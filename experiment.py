
import torch, numpy as np, os, json, time, warnings
from PIL import Image
import torch.nn.functional as F
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "C:/btp/dataset"
RESULTS_FILE = "C:/btp/results.json"

print(f"Device: {DEVICE}")

images, labels = [], []
for folder in sorted(os.listdir(DATASET_PATH)):
    fp = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(fp): continue
    for fname in os.listdir(fp):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            images.append(os.path.join(fp, fname))
            labels.append(folder)
print(f"Found {len(images)} images across {len(set(labels))} categories")

def evaluate(embeddings, labels):
    emb = torch.tensor(np.array(embeddings))
    emb = F.normalize(emb, dim=-1)
    sim = (emb @ emb.T).numpy()
    p1, p5, r5, mrr = [], [], [], []
    for i in range(len(labels)):
        s = sim[i].copy()
        s[i] = -999
        top5 = np.argsort(s)[::-1][:5]
        true = labels[i]
        if not any(labels[j] == true and j != i for j in range(len(labels))):
            continue
        p1.append(1.0 if labels[top5[0]] == true else 0.0)
        p5.append(sum(1 for idx in top5 if labels[idx] == true) / 5.0)
        r5.append(1.0 if any(labels[idx] == true for idx in top5) else 0.0)
        for rank, idx in enumerate(top5, 1):
            if labels[idx] == true:
                mrr.append(1.0/rank)
                break
        else:
            mrr.append(0.0)
    return {
        "Precision@1": round(float(np.mean(p1))*100, 2),
        "Precision@5": round(float(np.mean(p5))*100, 2),
        "Recall@5":    round(float(np.mean(r5))*100, 2),
        "MRR":         round(float(np.mean(mrr))*100, 2),
    }

results = {}

print("\n[1/4] Testing OpenCLIP...")
import open_clip
m1, _, p1 = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
m1 = m1.to(DEVICE).eval()
embs, t0 = [], time.time()
for path in images:
    img = p1(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embs.append(m1.encode_image(img).cpu().numpy()[0])
t = round((time.time()-t0)/len(images)*1000, 1)
r = evaluate(embs, labels)
r["inference_ms"] = t
r["text_search"] = "Yes"
r["embedding_dim"] = 512
results["OpenCLIP"] = r
print(f"Done! P@1={r['Precision@1']}% Speed={t}ms/img")
del m1
torch.cuda.empty_cache()

print("\n[2/4] Testing SigLIP...")
from transformers import AutoProcessor, AutoModel
proc2 = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
m2 = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(DEVICE).eval()
embs, t0 = [], time.time()
for path in images:
    inp = proc2(images=Image.open(path).convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out2 = m2.get_image_features(**inp)
        emb2 = out2.pooler_output if hasattr(out2, 'pooler_output') else out2.last_hidden_state[:,0,:]
        embs.append(emb2.cpu().numpy()[0])
t = round((time.time()-t0)/len(images)*1000, 1)
r = evaluate(embs, labels)
r["inference_ms"] = t
r["text_search"] = "Yes"
r["embedding_dim"] = 768
results["SigLIP"] = r
print(f"Done! P@1={r['Precision@1']}% Speed={t}ms/img")
del m2
torch.cuda.empty_cache()

print("\n[3/4] Testing DINOv2...")
from transformers import AutoImageProcessor, AutoModel
proc3 = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
m3 = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
embs, t0 = [], time.time()
for path in images:
    inp = proc3(images=Image.open(path).convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = m3(**inp)
        embs.append(out.last_hidden_state[:,0,:].cpu().numpy()[0])
t = round((time.time()-t0)/len(images)*1000, 1)
r = evaluate(embs, labels)
r["inference_ms"] = t
r["text_search"] = "No"
r["embedding_dim"] = 768
results["DINOv2"] = r
print(f"Done! P@1={r['Precision@1']}% Speed={t}ms/img")
del m3
torch.cuda.empty_cache()

print("\n[4/4] Testing BLIP-2 on CPU (30-45 mins)...")
from transformers import Blip2Processor, Blip2Model
proc4 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
m4 = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32).to("cpu").eval()
embs, t0 = [], time.time()
for i, path in enumerate(images):
    inp = proc4(images=Image.open(path).convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        embs.append(m4.get_image_features(**inp).pooler_output.cpu().numpy()[0])
    if (i+1) % 10 == 0:
        print(f"  Progress: {i+1}/{len(images)}")
t = round((time.time()-t0)/len(images)*1000, 1)
r = evaluate(embs, labels)
r["inference_ms"] = t
r["text_search"] = "Yes"
r["embedding_dim"] = 1408
results["BLIP-2"] = r
print(f"Done! P@1={r['Precision@1']}% Speed={t}ms/img")

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*65)
print("FINAL RESULTS TABLE")
print("="*65)
print(f"{'Model':<12} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'ms/img':>8} {'Text?':>6}")
print("-"*65)
for name, m in results.items():
    print(f"{name:<12} {m['Precision@1']:>5}% {m['Precision@5']:>5}% {m['Recall@5']:>5}% {m['MRR']:>5}% {m['inference_ms']:>7}ms {m['text_search']:>6}")
print("="*65)
print(f"Results saved to: {RESULTS_FILE}")
