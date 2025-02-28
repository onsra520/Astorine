import os, sys
from pathlib import Path
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

paths = {
    "processed": os.path.abspath(f"{project_root}/data/storage/processed"),
    "odata": os.path.abspath(f"{project_root}/data/storage/processed/final_cleaning.csv"),
    "config" : os.path.abspath(f"{project_root}/config/thresholds.json"),
    "models": os.path.abspath(f"{project_root}//models"),
    "encoder": os.path.abspath(f"{project_root}/models/encoder"),
    "scalers": os.path.abspath(f"{project_root}/models/decoder"),
    "hbm": os.path.abspath(f"{project_root}/models/compararison.pth"),
}

from nlp.extractor import ftdf
from training.hbmtrain import Hybrid_Training

odata = pd.read_csv(paths["odata"])

hbm = Hybrid_Training(train=False)
model = hbm.model
model.load_state_dict(
    torch.load(paths["hbm"], weights_only=False)
)
model.eval()

bias_mapping = {
    "gpu": {
        "gaming": 2.0,
        "3D simulations": 1.8,
        "high-performance": 1.5,
        "video editing": 1.7,
        "CAD": 1.3,
        "machine learning": 1.2,
        "animation": 1.7,
        "graphic design": 1.0,
        "virtual reality": 1.8,
        "professional rendering": 1.5,
        "augmented reality": 1.6,
        "crypto mining": 1.4,
        "real-time ray tracing": 1.8,
        "GPU accelerated rendering": 1.5,
        "streaming": 1.4,
        "deep learning": 1.4,
        "scientific visualization": 1.3,
        "virtual desktop": 1.2,
        "video encoding": 1.3,
        "VR development": 1.6,
        "AI processing": 1.3,
        "computational fluid dynamics": 1.3,
        "simulation acceleration": 1.4,
        "game development": 1.5,
        "3D modeling": 1.4,
        "shader processing": 1.3,
        "GPGPU": 1.5,
        "parallel processing": 1.5,
        "hardware acceleration": 1.5,
        "offline ray tracing": 1.4,
        "visual effects": 1.6,
        "post-production": 1.4,
        "real-time graphics": 1.5,
        "render farm": 1.3
    },
    "cpu": {
        "gaming": 1.0,
        "3D simulations": 1.2,
        "high-performance": 1.5,
        "software development": 1.0,
        "data analysis": 1.1,
        "rendering": 1.5,
        "multitasking": 1.4,
        "office work": 0.5,
        "virtualization": 1.3,
        "scientific computing": 1.2,
        "video editing": 1.3,
        "audio processing": 1.0,
        "database management": 1.0,
        "computational physics": 1.2,
        "compilation": 1.0,
        "batch processing": 1.2,
        "parallel computing": 1.3,
        "algorithmic trading": 1.1,
        "encryption": 1.2,
        "decryption": 1.2,
        "server applications": 1.3,
        "financial modeling": 1.2,
        "data mining": 1.3,
        "analytics": 1.1,
        "spreadsheet processing": 0.8,
        "web hosting": 1.0,
        "networking tasks": 1.0,
        "simulation modeling": 1.2,
        "real-time processing": 1.3,
        "compression": 1.1,
        "multimedia processing": 1.2,
        "script execution": 1.0,
        "data logging": 1.0
    },
    "ram": {
        "gaming": 1.0,
        "video editing": 1.5,
        "3D simulations": 1.3,
        "multitasking": 1.5,
        "rendering": 1.5,
        "CAD": 1.0,
        "animation": 1.2,
        "machine learning": 1.2,
        "virtual machines": 1.5,
        "large datasets": 1.2,
        "software development": 1.0,
        "content creation": 1.3,
        "multimedia": 1.2,
        "streaming": 1.1,
        "server applications": 1.0,
        "big data processing": 1.5,
        "memory caching": 1.3,
        "virtualized environments": 1.4,
        "real-time analytics": 1.3,
        "cloud computing": 1.2,
        "data streaming": 1.2,
        "augmented reality applications": 1.3,
        "video streaming": 1.2,
        "simulation of physics": 1.3,
        "multimedia editing": 1.3,
        "virtual reality editing": 1.3,
        "running multiple apps": 1.4,
        "browser tab management": 1.0,
        "enterprise applications": 1.2,
        "software emulation": 1.1,
        "data virtualization": 1.2,
        "concurrent processes": 1.3,
        "memory intensive tasks": 1.4,
        "high-speed data access": 1.3,
        "dynamic resource allocation": 1.2
    },
    "refresh rate": {
        "gaming": 2.0,
        "video editing": 1.0,
        "animation": 1.0,
        "3D simulations": 1.0,
        "creative": 1.0,
        "virtual reality": 1.8,
        "sports": 1.0,
        "motion clarity": 1.5,
        "fast-paced action": 1.2,
        "smooth scrolling": 1.0,
        "eSports": 1.8,
        "frame consistency": 1.2,
        "dynamic refresh": 1.0,
        "low latency": 1.5,
        "ultra-fast response": 1.2,
        "high frame rate": 2.0,
        "video fluidity": 1.0,
        "motion smoothing": 1.3,
        "display responsiveness": 1.2,
        "gameplay fluidity": 1.8,
        "reduced ghosting": 1.0,
        "adaptive sync": 1.7,
        "stutter-free": 1.0,
        "real-time feedback": 1.1,
        "frame interpolation": 1.3,
        "ultra-smooth": 1.4,
        "high refresh performance": 1.5,
        "cinematic fluidity": 1.1,
        "rapid update": 1.0
    },
    "resolution": {
        "design": 1.5,
        "creative": 1.5,
        "photo editing": 2.0,
        "video editing": 1.8,
        "graphic design": 1.5,
        "4k": 2.0,
        "ultra-high-definition": 1.8,
        "retina": 1.5,
        "digital art": 1.7,
        "CAD": 1.2,
        "high-definition": 1.5,
        "8k": 2.2,
        "full hd": 1.4,
        "high pixel density": 1.6,
        "QHD": 1.7,
        "WQHD": 1.7,
        "Super Retina": 1.6,
        "2k": 1.5,
        "HD+": 1.3,
        "pixel perfect": 1.8,
        "high clarity": 1.6,
        "crisp display": 1.4,
        "vivid resolution": 1.4,
        "sharp detail": 1.5,
        "visual fidelity": 1.6,
        "ultra-sharp": 1.5,
        "true color": 1.5,
        "precision imaging": 1.4,
        "retina quality": 1.5,
        "maximum resolution": 1.5,
        "dynamic resolution": 1.3,
        "image clarity": 1.4,
        "high-density pixels": 1.6,
        "optimized resolution": 1.3
    },
    "display type": {
        "IPS": 1.5,
        "OLED": 1.8,
        "color-accurate": 2.0,
        "wide color gamut": 2.0,
        "touchscreen": 1.0,
        "matte": 1.2,
        "glossy": 1.0,
        "professional grading": 2.5,
        "HDR": 2.0,
        "anti-glare": 1.5,
        "LED-backlit": 1.2,
        "QLED": 1.8,
        "microLED": 1.8,
        "quantum dot": 1.8,
        "flicker-free": 1.3,
        "energy efficient": 1.2,
        "flexible display": 1.0,
        "curved": 1.2,
        "flat panel": 1.0,
        "dual-mode": 1.1,
        "high brightness": 1.4,
        "color uniformity": 1.5,
        "smooth gradient": 1.3,
        "adaptive brightness": 1.2,
        "local dimming": 1.4,
        "full-array": 1.3,
        "edge-lit": 1.2,
        "IPS-level": 1.4,
        "sRGB standard": 1.5,
        "HDR10": 2.0,
        "Dolby Vision": 2.0,
        "ambient light sensor": 1.0
    },
    "screen size": {
        "design": 1.5,
        "creative": 1.5,
        "large": 1.5,
        "ultra-wide": 2.0,
        "portable": 0.8,
        "multitasking": 1.5,
        "video editing": 1.5,
        "immersive experience": 1.3,
        "split-screen": 1.0,
        "widescreen": 1.5,
        "multi-monitor": 1.3,
        "extra large": 1.8,
        "cinematic": 1.6,
        "compact": 0.8,
        "full screen": 1.3,
        "bezel-less": 1.4,
        "borderless": 1.4,
        "ultra-portable": 0.9,
        "desktop replacement": 1.6,
        "multi-display": 1.3,
        "wide aspect": 1.4,
        "panoramic": 1.5,
        "immersive widescreen": 1.6,
        "large format": 1.5,
        "high-resolution screen": 1.4,
        "expanded view": 1.3,
        "maximized display": 1.2,
        "enterprise monitor": 1.1,
        "ultra-large": 1.7,
        "portable compact": 0.8,
        "gaming monitor": 1.5,
        "adjustable size": 1.0,
        "dynamic screen": 1.0,
        "versatile display": 1.2
    }
}

def ex_bias(query: str = None) -> dict:
    bias_result = {key: 1 for key in bias_mapping.keys()}
    if query:
        query_lower = query.lower()
        for key, mapping in bias_mapping.items():
            for keyword, weight in mapping.items():
                if keyword in query_lower:
                    bias_result[key] += weight
    return bias_result

def searching(query: str=None) -> pd.DataFrame:
    sorted_df = hbm.df.copy()
    if sorted_df.empty:
        return sorted_df
    
    numerical_input = torch.tensor(sorted_df[hbm.numerical_columns].values, dtype=torch.float, device=hbm.device)
    categorical_input = torch.tensor(sorted_df[hbm.categorical_columns].values, dtype=torch.long, device=hbm.device)

    model.eval()
    with torch.no_grad():
        outputs = model(numerical_input, categorical_input)
        predicted_scores = outputs.squeeze().cpu().numpy()
    
    sorted_df["PREDICTED_SCORE"] = predicted_scores
    sorted_filtered = sorted_df.sort_values(
        by=["PREDICTED_SCORE", "CPU PASSMARK RESULT", "GPU PASSMARK (G3D) RESULT"],
        ascending=[False, False, False]
    )
    sorted_filtered.columns = [col.lower() for col in sorted_filtered.columns]
    if query is None:
        return sorted_filtered["device"]
    
    extract = ftdf(query=str(query), weights=ex_bias(str(query)), thresholds_name="threshold_1")
    result = pd.merge(sorted_filtered, extract, on='device', how='right', sort=False)
    
    return result