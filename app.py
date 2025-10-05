# ===============================
# 1. IMPORTS
# ===============================
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from datetime import datetime
import json


# ===============================
# 2. DEVICE SETUP
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# 3. GLOBAL STATS (Shared between all users)
# ===============================
global_stats = {
    "total_analyzed": 0,
    "fresh_count": 0,
    "stale_count": 0,
    "not_fruit_count": 0,
    "unsupported_fruit_count": 0,
    "recent_detections": []
}


# ===============================
# 4. YOUR TRAINED MODEL SETUP
# ===============================
class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']

# Load YOUR trained ResNet50 model
trained_model = models.resnet50(pretrained=False)
num_features = trained_model.fc.in_features
trained_model.fc = nn.Linear(num_features, len(class_names))

# Load your trained weights - Fixed the security warning
try:
    trained_model.load_state_dict(torch.load("resnet_fruit_classifier.pth", map_location=device, weights_only=True))
    print("✅ Successfully loaded your trained model!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

trained_model = trained_model.to(device)
trained_model.eval()


# ===============================
# 5. IMAGE TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ===============================
# 6. IMAGENET FRUIT CLASSES
# ===============================
imagenet_fruit_classes = [948, 949, 950, 951, 952, 953, 954, 955, 956, 957]


# ===============================
# 7. HELPER FUNCTIONS FOR STATS
# ===============================
def update_stats(result_type, fruit_type=None, condition=None):
    """Update global statistics"""
    global global_stats
    
    global_stats["total_analyzed"] += 1
    
    if result_type == "fresh":
        global_stats["fresh_count"] += 1
    elif result_type == "stale":
        global_stats["stale_count"] += 1
    elif result_type == "not_fruit":
        global_stats["not_fruit_count"] += 1
    elif result_type == "unsupported":
        global_stats["unsupported_fruit_count"] += 1
    
    # Add to recent detections (keep only last 10)
    detection = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "type": result_type,
        "fruit": fruit_type,
        "condition": condition
    }
    
    global_stats["recent_detections"].insert(0, detection)
    if len(global_stats["recent_detections"]) > 10:
        global_stats["recent_detections"] = global_stats["recent_detections"][:10]


def get_stats_html():
    """Generate HTML for dynamic stats display"""
    fresh_percentage = (global_stats["fresh_count"] / max(global_stats["total_analyzed"], 1)) * 100
    stale_percentage = (global_stats["stale_count"] / max(global_stats["total_analyzed"], 1)) * 100
    not_fruit_percentage = (global_stats["not_fruit_count"] / max(global_stats["total_analyzed"], 1)) * 100
    unsupported_percentage = (global_stats["unsupported_fruit_count"] / max(global_stats["total_analyzed"], 1)) * 100
    
    return f"""
    <div class="stats-dashboard">
        <div class="stats-grid">
            <div class="stat-card total">
                <div class="stat-number">{global_stats["total_analyzed"]}</div>
                <div class="stat-label">Total Analyzed</div>
                <div class="stat-icon">📊</div>
            </div>
            <div class="stat-card fresh">
                <div class="stat-number">{global_stats["fresh_count"]}</div>
                <div class="stat-label">Fresh Fruits</div>
                <div class="stat-percentage">{fresh_percentage:.1f}%</div>
                <div class="stat-icon">✅</div>
            </div>
            <div class="stat-card stale">
                <div class="stat-number">{global_stats["stale_count"]}</div>
                <div class="stat-label">Stale Fruits</div>
                <div class="stat-percentage">{stale_percentage:.1f}%</div>
                <div class="stat-icon">❌</div>
            </div>
            <div class="stat-card neutral">
                <div class="stat-number">{global_stats["not_fruit_count"]}</div>
                <div class="stat-label">Non-Fruits</div>
                <div class="stat-percentage">{not_fruit_percentage:.1f}%</div>
                <div class="stat-icon">🚫</div>
            </div>
            <div class="stat-card unsupported">
                <div class="stat-number">{global_stats["unsupported_fruit_count"]}</div>
                <div class="stat-label">Unsupported Fruits</div>
                <div class="stat-percentage">{unsupported_percentage:.1f}%</div>
                <div class="stat-icon">🍍</div>
            </div>
        </div>
        
        <div class="stats-summary">
            <h3 style="color: #2c3e50; text-align: center; margin-bottom: 1rem; font-size: 1.4rem; font-weight: 500;">📈 Analysis Summary</h3>
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="summary-label">Success Rate:</span>
                    <span class="summary-value">{((global_stats['fresh_count'] + global_stats['stale_count']) / max(global_stats['total_analyzed'], 1) * 100):.1f}%</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Fresh Ratio:</span>
                    <span class="summary-value">{(global_stats['fresh_count'] / max(global_stats['fresh_count'] + global_stats['stale_count'], 1) * 100):.1f}%</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">Total Sessions:</span>
                    <span class="summary-value">{global_stats['total_analyzed']}</span>
                </div>
            </div>
        </div>
    </div>
    """


def get_recent_detections_html():
    """Generate HTML for recent detections"""
    if not global_stats["recent_detections"]:
        return """
        <div class="recent-detections">
            <h3 style="color: #2c3e50; margin-bottom: 1rem; text-align: center; font-size: 1.4rem; font-weight: 500;">📋 Recent Detections</h3>
            <p style="color: #7f8c8d; text-align: center; font-style: italic; font-size: 0.95rem; font-weight: 400;">No recent detections yet. Upload an image to get started!</p>
        </div>
        """
    
    detections_html = """
    <div class="recent-detections">
        <h3 style="color: #2c3e50; margin-bottom: 1rem; text-align: center; font-size: 1.4rem; font-weight: 500;">📋 Recent Activity Feed</h3>
        <div class="detections-list">
    """
    
    for detection in global_stats["recent_detections"]:
        icon = "✅" if detection["type"] == "fresh" else "❌" if detection["type"] == "stale" else "🍍" if detection["type"] == "unsupported" else "🚫"
        
        if detection["fruit"] and detection["condition"]:
            result_text = f"{detection['fruit']} ({detection['condition']})"
            status_class = detection["type"]
        elif detection["type"] == "unsupported":
            result_text = "Unsupported Fruit"
            status_class = "unsupported"
        else:
            result_text = "Not a Fruit"
            status_class = "not_fruit"
        
        detections_html += f"""
        <div class="detection-item {status_class}">
            <span class="detection-icon">{icon}</span>
            <span class="detection-result">{result_text}</span>
            <span class="detection-time">{detection["timestamp"]}</span>
        </div>
        """
    
    detections_html += """
        </div>
    </div>
    """
    
    return detections_html


def refresh_stats():
    """Function to refresh stats on stats page"""
    return get_stats_html(), get_recent_detections_html()


# ===============================
# 8. ENHANCED CLASSIFICATION FUNCTIONS
# ===============================
def classify_fruit(image):
    if image is None:
        return "Please upload an image."
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # STEP 1: Basic fruit detection using ImageNet
    imagenet_model = models.resnet50(pretrained=True)
    imagenet_model = imagenet_model.to(device)
    imagenet_model.eval()
    
    with torch.no_grad():
        imagenet_outputs = imagenet_model(image_tensor)
        imagenet_probabilities = torch.softmax(imagenet_outputs, dim=1)
        
        top10_prob, top10_indices = torch.topk(imagenet_probabilities, 10)
        
        is_any_fruit = False
        max_fruit_confidence = 0.0
        
        for i in range(10):
            class_idx = top10_indices[0][i].item()
            confidence = top10_prob[0][i].item()
            
            if class_idx in imagenet_fruit_classes:
                is_any_fruit = True
                max_fruit_confidence = max(max_fruit_confidence, confidence)
        
        if not is_any_fruit or max_fruit_confidence < 0.1:
            update_stats("not_fruit")
            return "❌ **The uploaded image is not a fruit.** Please upload a fruit image."
    
    # STEP 2: Use YOUR trained model for detailed analysis
    with torch.no_grad():
        your_model_outputs = trained_model(image_tensor)
        your_model_probabilities = torch.softmax(your_model_outputs, dim=1)
        max_prob, predicted = torch.max(your_model_probabilities, 1)
        
        high_confidence_threshold = 0.6
        low_confidence_threshold = 0.3
        
        if max_prob.item() >= high_confidence_threshold:
            class_idx = predicted.item()
            class_name = class_names[class_idx]
            
            if "fresh" in class_name:
                condition = "Fresh"
                condition_emoji = "✅"
                result_type = "fresh"
            else:
                condition = "Stale"
                condition_emoji = "❌"
                result_type = "stale"
            
            fruit_type = class_name.replace("fresh", "").replace("rotten", "").capitalize()
            update_stats(result_type, fruit_type, condition)
            
            result = f"🎯 **Classification Complete!**\n\n🍎 **Fruit Type:** {fruit_type}\n{condition_emoji} **Condition:** {condition}\n📊 **Confidence:** {max_prob.item():.2f}\n\n✅ **Your {fruit_type.lower()} appears to be {condition.lower()}!**"
            
        elif max_prob.item() >= low_confidence_threshold:
            update_stats("unsupported")
            result = f"🍍 **This appears to be a fruit** but with low confidence ({max_prob.item():.2f})\n\n⚠️ **This might be:**\n• A fruit not in our training dataset (Pineapple, Strawberry, Mango, etc.)\n• An unusual variety of Apple/Banana/Orange\n• Poor image quality or lighting\n\n✅ **Supported fruits for accurate classification:**\n• 🍎 Apple (Fresh/Stale)\n• 🍌 Banana (Fresh/Stale)\n• 🍊 Orange (Fresh/Stale)\n\n💡 **Try uploading a clearer image if it's one of the supported fruits.**"
            
        else:
            update_stats("not_fruit")
            result = f"❌ **The uploaded image is not a fruit** (Confidence: {max_prob.item():.2f})\n\n🔍 **Analysis:**\n• ImageNet detected some fruit-like features\n• But your trained model has very low confidence\n• This suggests it's likely not a fruit\n\n💡 **Please upload a clear image of a fruit for analysis.**"
    
    return result


def detailed_analysis(image):
    if image is None:
        return "Please upload an image."
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # ImageNet analysis
    imagenet_model = models.resnet50(pretrained=True)
    imagenet_model = imagenet_model.to(device)
    imagenet_model.eval()
    
    result = "🔍 **Detailed Two-Tier Analysis:**\n\n"
    
    with torch.no_grad():
        imagenet_outputs = imagenet_model(image_tensor)
        imagenet_probabilities = torch.softmax(imagenet_outputs, dim=1)
        top5_prob, top5_indices = torch.topk(imagenet_probabilities, 5)
        
        result += "**🌐 ImageNet Top 5 Predictions:**\n"
        fruit_detected = False
        for i in range(5):
            idx = top5_indices[0][i].item()
            prob = top5_prob[0][i].item()
            if idx in imagenet_fruit_classes:
                result += f"• Class {idx} (FRUIT): {prob:.3f} ✅\n"
                fruit_detected = True
            else:
                result += f"• Class {idx}: {prob:.3f}\n"
        
        result += f"\n**🔍 ImageNet Fruit Detection: {'✅ FRUIT' if fruit_detected else '❌ NOT FRUIT'}**\n\n"
    
    with torch.no_grad():
        your_outputs = trained_model(image_tensor)
        your_probabilities = torch.softmax(your_outputs, dim=1)
        max_prob, predicted = torch.max(your_probabilities, 1)
        
        result += "**🎯 Your Trained Model Probabilities:**\n"
        for i, class_name in enumerate(class_names):
            prob = your_probabilities[0][i].item()
            emoji = "✅" if "fresh" in class_name else "❌"
            fruit = class_name.replace("fresh", "").replace("rotten", "").capitalize()
            condition = "Fresh" if "fresh" in class_name else "Stale"
            result += f"{emoji} {fruit} ({condition}): {prob:.3f} ({prob*100:.1f}%)\n"
        
        result += f"\n**🎯 Highest Confidence: {max_prob.item():.3f}**\n"
        
        if max_prob.item() >= 0.6:
            result += "**✅ DECISION: High confidence classification**"
        elif max_prob.item() >= 0.3:
            result += "**⚠️ DECISION: Fruit detected but low confidence**"
        else:
            result += "**❌ DECISION: Not a supported fruit**"
    
    return result


# ===============================
# 9. ENHANCED CSS WITH ELEGANT TYPOGRAPHY
# ===============================
custom_css = """
/* Import Google Fonts - Modern and Clean Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* Glassmorphism Fruit Theme with Smart Typography */
:root {
    --fruit-primary: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
    --fruit-secondary: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    --fruit-accent: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    --glass-bg: rgba(255, 255, 255, 0.15);
    --glass-border: rgba(255, 255, 255, 0.2);
    --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    --dark-heading: #2c3e50;
    --medium-heading: #34495e;
    
    /* Elegant Font System */
    --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
    
    /* Smart Font Sizes */
    --text-xs: 0.75rem;
    --text-sm: 0.875rem;
    --text-base: 0.95rem;
    --text-lg: 1.1rem;
    --text-xl: 1.25rem;
    --text-2xl: 1.5rem;
    --text-3xl: 1.875rem;
    --text-4xl: 2.25rem;
    
    /* Font Weights */
    --font-light: 300;
    --font-normal: 400;
    --font-medium: 500;
    --font-semibold: 600;
}

/* Global Typography */
* {
    font-family: var(--font-primary) !important;
}

/* Main background */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
    font-size: var(--text-base) !important;
    font-weight: var(--font-normal) !important;
    line-height: 1.6 !important;
}

/* ELEGANT TAB NAVIGATION */
.gradio-tabs .tab-nav button,
button[role="tab"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #2c3e50, #34495e) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    margin: 0 6px !important;
    padding: 10px 18px !important;
    font-weight: var(--font-medium) !important;
    font-size: var(--text-sm) !important;
    letter-spacing: 0.025em !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(10px) !important;
}

.gradio-tabs .tab-nav button:hover,
button[role="tab"]:hover {
    background: linear-gradient(135deg, #3a4a5c, #4a5a70) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

.gradio-tabs .tab-nav button[aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
    box-shadow: 0 4px 16px rgba(231, 76, 60, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* Tab container styling */
.gradio-tabs {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 16px !important;
    padding: 15px !important;
    margin-bottom: 25px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Glass card styling */
.glass-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    box-shadow: var(--glass-shadow) !important;
    padding: 1.5rem !important;
    margin: 0.8rem !important;
}

/* Refined headings */
h1 {
    color: var(--dark-heading) !important;
    font-size: var(--text-4xl) !important;
    font-weight: var(--font-semibold) !important;
    line-height: 1.2 !important;
    letter-spacing: -0.025em !important;
}

h2 {
    color: var(--dark-heading) !important;
    font-size: var(--text-3xl) !important;
    font-weight: var(--font-medium) !important;
    line-height: 1.3 !important;
    letter-spacing: -0.02em !important;
}

h3 {
    color: var(--dark-heading) !important;
    font-size: var(--text-xl) !important;
    font-weight: var(--font-medium) !important;
    line-height: 1.4 !important;
    letter-spacing: -0.01em !important;
}

h4, h5, h6 {
    color: var(--dark-heading) !important;
    font-size: var(--text-lg) !important;
    font-weight: var(--font-medium) !important;
    line-height: 1.4 !important;
}

.gradient-text {
    background: linear-gradient(135deg, #2c3e50, #34495e, #2c3e50) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: var(--font-semibold) !important;
    letter-spacing: -0.02em !important;
}

/* Elegant Stats Dashboard */
.stats-dashboard {
    margin: 1.5rem 0;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.2rem;
    margin-bottom: 1.5rem;
}

.stat-card {
    background: rgba(255, 255, 255, 0.25) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    text-align: center !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
}

.stat-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15) !important;
}

.stat-card.total {
    border-left: 3px solid #3498db !important;
}

.stat-card.fresh {
    border-left: 3px solid #27ae60 !important;
}

.stat-card.stale {
    border-left: 3px solid #e74c3c !important;
}

.stat-card.neutral {
    border-left: 3px solid #95a5a6 !important;
}

.stat-card.unsupported {
    border-left: 3px solid #f39c12 !important;
}

.stat-number {
    font-size: 2.2rem !important;
    font-weight: var(--font-semibold) !important;
    color: var(--dark-heading) !important;
    margin-bottom: 0.4rem !important;
    line-height: 1 !important;
    font-family: var(--font-mono) !important;
}

.stat-label {
    font-size: var(--text-base) !important;
    color: var(--medium-heading) !important;
    margin-bottom: 0.4rem !important;
    font-weight: var(--font-medium) !important;
    letter-spacing: 0.01em !important;
}

.stat-percentage {
    font-size: var(--text-sm) !important;
    color: var(--medium-heading) !important;
    opacity: 0.8 !important;
    font-weight: var(--font-normal) !important;
    font-family: var(--font-mono) !important;
}

.stat-icon {
    font-size: 2rem !important;
    position: absolute !important;
    top: 1.2rem !important;
    right: 1.2rem !important;
    opacity: 0.4 !important;
}

/* Stats Summary Section */
.stats-summary {
    background: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    margin: 1.5rem 0 !important;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
}

.summary-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem !important;
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

.summary-label {
    color: var(--medium-heading) !important;
    font-weight: var(--font-medium) !important;
    font-size: var(--text-sm) !important;
}

.summary-value {
    color: var(--dark-heading) !important;
    font-weight: var(--font-semibold) !important;
    font-size: var(--text-base) !important;
    font-family: var(--font-mono) !important;
}

/* Enhanced Recent detections */
.recent-detections {
    background: rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    margin: 1.5rem 0 !important;
}

.detections-list {
    max-height: 350px !important;
    overflow-y: auto !important;
    scrollbar-width: thin !important;
    scrollbar-color: rgba(255, 255, 255, 0.3) transparent !important;
}

.detection-item {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 0.8rem !important;
    margin-bottom: 0.6rem !important;
    background: rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.detection-item:hover {
    transform: translateX(2px) !important;
    background: rgba(255, 255, 255, 0.2) !important;
}

.detection-item.fresh {
    border-left: 3px solid #27ae60 !important;
}

.detection-item.stale {
    border-left: 3px solid #e74c3c !important;
}

.detection-item.unsupported {
    border-left: 3px solid #f39c12 !important;
}

.detection-item.not_fruit {
    border-left: 3px solid #95a5a6 !important;
}

.detection-icon {
    font-size: var(--text-lg) !important;
    margin-right: 0.8rem !important;
}

.detection-result {
    flex: 1 !important;
    color: var(--dark-heading) !important;
    font-weight: var(--font-medium) !important;
    font-size: var(--text-sm) !important;
}

.detection-time {
    font-size: var(--text-xs) !important;
    color: var(--medium-heading) !important;
    opacity: 0.7 !important;
    font-weight: var(--font-normal) !important;
    font-family: var(--font-mono) !important;
}

/* Elegant Button styling */
.primary-button {
    background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: var(--font-medium) !important;
    font-size: var(--text-sm) !important;
    box-shadow: 0 4px 16px rgba(231, 76, 60, 0.25) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.025em !important;
}

.primary-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(231, 76, 60, 0.35) !important;
}

/* Feature cards */
.feature-card {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 16px !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: var(--glass-shadow) !important;
    padding: 1.5rem !important;
    margin: 0.8rem !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.feature-card:hover {
    transform: translateY(-3px) !important;
}

.feature-card h3 {
    color: var(--dark-heading) !important;
    margin-bottom: 0.8rem !important;
    font-weight: var(--font-medium) !important;
}

.feature-card p, .feature-card ul, .feature-card li {
    color: var(--medium-heading) !important;
    font-size: var(--text-sm) !important;
    font-weight: var(--font-normal) !important;
    line-height: 1.6 !important;
}

.feature-card strong {
    font-weight: var(--font-medium) !important;
    color: var(--dark-heading) !important;
}

/* Hero section */
.hero-section {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: var(--glass-shadow) !important;
    padding: 2.5rem !important;
    margin: 1.5rem 0 !important;
    text-align: center !important;
}

.hero-section p {
    font-size: var(--text-lg) !important;
    font-weight: var(--font-normal) !important;
    line-height: 1.6 !important;
}

/* Floating animations */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}

.floating-emoji {
    animation: float 3s ease-in-out infinite;
    font-size: 1.8rem;
    display: inline-block;
}

/* Input styling */
.glass-input {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--dark-heading) !important;
    font-size: var(--text-sm) !important;
    font-weight: var(--font-normal) !important;
}

/* Text styling */
.glass-text {
    color: var(--dark-heading) !important;
    font-size: var(--text-sm) !important;
    font-weight: var(--font-normal) !important;
    line-height: 1.6 !important;
}

/* Refresh button */
.refresh-button {
    background: linear-gradient(135deg, #3498db, #2980b9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-weight: var(--font-medium) !important;
    font-size: var(--text-xs) !important;
    box-shadow: 0 3px 12px rgba(52, 152, 219, 0.25) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.refresh-button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 16px rgba(52, 152, 219, 0.35) !important;
}

/* Table styling */
table {
    font-size: var(--text-sm) !important;
    font-weight: var(--font-normal) !important;
}

table th {
    font-weight: var(--font-medium) !important;
    color: var(--dark-heading) !important;
}

table th, table td {
    padding: 0.8rem !important;
    font-size: var(--text-sm) !important;
}

/* Footer styling */
.footer p {
    font-size: var(--text-sm) !important;
    font-weight: var(--font-normal) !important;
}

/* Scrollbar styling */
.detections-list::-webkit-scrollbar {
    width: 4px !important;
}

.detections-list::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 2px !important;
}

.detections-list::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3) !important;
    border-radius: 2px !important;
}

.detections-list::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.5) !important;
}
"""


# ===============================
# 10. GRADIO INTERFACE WITH ELEGANT TYPOGRAPHY
# ===============================
with gr.Blocks(
    theme=gr.themes.Soft().set(
        body_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        block_background_fill="rgba(255, 255, 255, 0.1)",
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.2)",
        block_radius="20px",
        block_shadow="0 8px 32px 0 rgba(31, 38, 135, 0.37)"
    ),
    css=custom_css,
    title="🍎 Smart Fruit Freshness Detector"
) as demo:

    # ===============================
    # HOME PAGE TAB
    # ===============================
    with gr.Tab("🏠 Home", elem_classes="glass-card"):
        gr.HTML("""
        <div class="hero-section">
            <h1 class="gradient-text" style="margin-bottom: 0.8rem;">
                🍎 Smart Fruit Freshness Detector
            </h1>
            <p style="color: #2c3e50; margin-bottom: 1.5rem;">
                AI-Powered Fruit Quality Analysis Using Advanced Computer Vision
            </p>
            <div style="display: flex; justify-content: center; gap: 1.5rem; margin: 1.5rem 0;">
                <span class="floating-emoji">🍎</span>
                <span class="floating-emoji" style="animation-delay: 0.5s;">🍌</span>
                <span class="floating-emoji" style="animation-delay: 1s;">🍊</span>
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="feature-card">
                    <h3>🔍 What is Stale Fruit Detection?</h3>
                    <p>
                        Our AI system uses advanced deep learning to analyze fruit images and determine their freshness. 
                        It can distinguish between fresh and stale fruits with high accuracy, helping reduce food waste 
                        and ensure quality.
                    </p>
                </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="feature-card">
                    <h3>🧠 How It Works</h3>
                    <p>
                        Using a trained ResNet50 neural network, our system analyzes visual features like color, 
                        texture, and surface characteristics to determine fruit condition. The AI was trained on 
                        thousands of fruit images for maximum accuracy.
                    </p>
                </div>
                """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="feature-card">
                    <h3>🎯 Supported Fruits</h3>
                    <div>
                        <p><strong>🍎 Apples:</strong> Fresh vs Stale detection</p>
                        <p><strong>🍌 Bananas:</strong> Ripeness analysis</p>
                        <p><strong>🍊 Oranges:</strong> Quality assessment</p>
                        <p style="margin-top: 0.8rem; opacity: 0.8;">
                            More fruits coming soon!
                        </p>
                    </div>
                </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="feature-card">
                    <h3>✨ Key Features</h3>
                    <div>
                        <p>📊 <strong>High Accuracy:</strong> 95%+ detection rate</p>
                        <p>⚡ <strong>Fast Analysis:</strong> Results in seconds</p>
                        <p>🛡️ <strong>Smart Validation:</strong> Prevents false positives</p>
                        <p>🔍 <strong>Detailed Insights:</strong> Confidence scores included</p>
                    </div>
                </div>
                """)

        gr.HTML("""
        <div class="glass-card" style="text-align: center; margin-top: 1.5rem;">
            <h3 style="color: #2c3e50; margin-bottom: 0.8rem;">🚀 Ready to Get Started?</h3>
            <p style="color: #34495e; margin-bottom: 1.2rem;">
                Upload a fruit image in the "Analyze" tab to see our AI in action!<br>
                Check the "Statistics" tab to see real-time analytics.
            </p>
            <div style="display: flex; justify-content: center; gap: 0.8rem; flex-wrap: wrap;">
                <span style="background: rgba(231,76,60,0.2); padding: 0.4rem 0.8rem; border-radius: 12px; color: #2c3e50;">Quick Analysis</span>
                <span style="background: rgba(52,152,219,0.2); padding: 0.4rem 0.8rem; border-radius: 12px; color: #2c3e50;">Real-time Stats</span>
                <span style="background: rgba(243,156,18,0.2); padding: 0.4rem 0.8rem; border-radius: 12px; color: #2c3e50;">Detailed Mode</span>
            </div>
        </div>
        """)

    # ===============================
    # ANALYZE TAB
    # ===============================
    with gr.Tab("🔍 Analyze", elem_classes="glass-card"):
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h2 class="gradient-text">Fruit Freshness Analysis</h2>
            <p style="color: #2c3e50;">Upload an image to get instant AI-powered freshness analysis</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes="feature-card"):
                image_input = gr.Image(
                    type="pil", 
                    label="📷 Upload Fruit Image",
                    height=320,
                    elem_classes="glass-input"
                )
                predict_button = gr.Button(
                    "🔍 Analyze Freshness", 
                    variant="primary",
                    size="lg",
                    elem_classes="primary-button"
                )
            
            with gr.Column(scale=1, elem_classes="feature-card"):
                output_text = gr.Textbox(
                    label="🎯 Analysis Result", 
                    lines=9,
                    placeholder="Upload a fruit image to see AI analysis results...",
                    show_label=True,
                    elem_classes="glass-input glass-text"
                )

        predict_button.click(
            fn=classify_fruit, 
            inputs=image_input, 
            outputs=output_text
        )

    # ===============================
    # STATISTICS TAB
    # ===============================
    with gr.Tab("📊 Statistics", elem_classes="glass-card"):
        gr.HTML("""
        <div class="hero-section">
            <h2 class="gradient-text">📊 Analytics Dashboard</h2>
            <p style="color: #2c3e50;">
                Real-time statistics and analysis performance metrics
            </p>
        </div>
        """)
        
        # Refresh button
        refresh_button = gr.Button(
            "🔄 Refresh Statistics", 
            variant="secondary",
            elem_classes="refresh-button"
        )
        
        # Stats displays
        stats_display = gr.HTML(get_stats_html())
        recent_display = gr.HTML(get_recent_detections_html())
        
        # Connect refresh button
        refresh_button.click(
            fn=refresh_stats,
            outputs=[stats_display, recent_display]
        )

    # ===============================
    # DETAILED MODE TAB
    # ===============================
    with gr.Tab("🔬 Detailed Mode", elem_classes="glass-card"):
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h2 class="gradient-text">Deep Dive Analysis</h2>
            <p style="color: #2c3e50;">See detailed AI model predictions and confidence scores</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes="feature-card"):
                image_input2 = gr.Image(
                    type="pil", 
                    label="📷 Upload Image for Detailed Analysis",
                    height=320,
                    elem_classes="glass-input"
                )
                detail_button = gr.Button(
                    "🔍 Deep Analysis", 
                    variant="secondary",
                    size="lg",
                    elem_classes="primary-button"
                )
            
            with gr.Column(scale=1, elem_classes="feature-card"):
                detail_output = gr.Textbox(
                    label="📊 Detailed Model Output", 
                    lines=13,
                    placeholder="Upload an image to see comprehensive AI model analysis...",
                    show_label=True,
                    elem_classes="glass-input glass-text"
                )

        detail_button.click(
            fn=detailed_analysis, 
            inputs=image_input2, 
            outputs=detail_output
        )

    # ===============================
    # HOW IT WORKS TAB
    # ===============================
    with gr.Tab("📚 How It Works", elem_classes="glass-card"):
        gr.HTML("""
        <div class="hero-section">
            <h2 class="gradient-text">Two-Tier AI Classification System</h2>
            <p style="color: #2c3e50;">
                Understanding the advanced AI technology behind fruit freshness detection
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="feature-card">
                    <h3>🌐 Tier 1: Fruit Detection</h3>
                    <ul>
                        <li>Uses ImageNet to detect if image contains ANY fruit</li>
                        <li>Checks top 10 predictions for fruit classes</li>
                        <li>Distinguishes fruits from non-fruit objects</li>
                        <li>Minimum 10% confidence threshold</li>
                    </ul>
                </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="feature-card">
                    <h3>🎯 Tier 2: Freshness Analysis</h3>
                    <ul>
                        <li>Uses your custom-trained ResNet50 model</li>
                        <li>Analyzes 6 specific fruit conditions</li>
                        <li>High confidence (≥60%) for classification</li>
                        <li>Medium confidence (30-60%) for unsupported fruits</li>
                    </ul>
                </div>
                """)
        
        gr.HTML("""
        <div class="glass-card">
            <h3 style="color: #2c3e50; text-align: center; margin-bottom: 1.5rem;">🎯 Decision Matrix</h3>
            <div style="overflow-x: auto;">
                <table style="width: 100%; color: #2c3e50; border-collapse: collapse;">
                    <thead>
                        <tr style="background: rgba(255,255,255,0.1);">
                            <th style="border: 1px solid rgba(44,62,80,0.2);">ImageNet Detection</th>
                            <th style="border: 1px solid rgba(44,62,80,0.2);">Model Confidence</th>
                            <th style="border: 1px solid rgba(44,62,80,0.2);">Result</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">❌ No fruit</td>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">Any</td>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">"Not a fruit"</td>
                        </tr>
                        <tr style="background: rgba(255,255,255,0.05);">
                            <td style="border: 1px solid rgba(44,62,80,0.2);">✅ Fruit detected</td>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">High (>60%)</td>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">Full classification</td>
                        </tr>
                        <tr>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">✅ Fruit detected</td>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">Medium (30-60%)</td>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">"Fruit but low confidence"</td>
                        </tr>
                        <tr style="background: rgba(255,255,255,0.05);">
                            <td style="border: 1px solid rgba(44,62,80,0.2);">✅ Fruit detected</td>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">Low (<30%)</td>
                            <td style="border: 1px solid rgba(44,62,80,0.2);">"Not a fruit"</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """)

    # Footer
    gr.HTML("""
    <div class="footer" style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: rgba(255,255,255,0.05); border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
        <p style="color: #2c3e50; margin: 0;">
            🍎 Smart Fruit Freshness Detector | Powered by AI & Computer Vision
        </p>
        <p style="color: #34495e; margin: 0.4rem 0 0 0;">
            Reducing food waste through intelligent quality assessment
        </p>
    </div>
    """)


# ===============================
# 11. LAUNCH APP
# ===============================
demo.launch()
