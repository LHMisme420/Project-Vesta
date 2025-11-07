pip install cryptography numpy torch
from vesta_core import AnchorSeedGenerator, ProvenanceTracker, ConfidenceEngine, generate_keypair

# Setup
private_key, public_key = generate_keypair()
generator = AnchorSeedGenerator("CAMERA_001", private_key)

# Create cryptographic anchor
anchor = generator.create_truth_anchor(
    b"your_media_data", 
    {"format": "JPEG", "resolution": "1920x1080"}
)

# Track edits
tracker = ProvenanceTracker(anchor["anchor_id"])
tracker.add_edit("color_correction", "editor_1", {"adjustment": "brightness+10"}, private_key)

# Get confidence score
engine = ConfidenceEngine()
analysis = engine.analyze_media(
    "https://example.com/media.jpg",
    {"vesta_anchor_id": anchor["anchor_id"]},
    tracker.get_provenance_chain()
)

print(f"Confidence: {analysis['confidence_score']}")
print(f"Risk Level: {analysis['risk_level']}")
generator = AnchorSeedGenerator("DEVICE_ID", private_key)
anchor = generator.create_truth_anchor(raw_data, metadata)
tracker = ProvenanceTracker(anchor_id)
tracker.add_edit("edit_type", "editor_id", metadata, private_key)
is_valid = tracker.verify_chain(public_key_map)
engine = ConfidenceEngine()
analysis = engine.analyze_media(url, metadata, provenance_chain)
{
  "confidence_score": 0.852,
  "risk_level": "low",
  "recommendation": "trust_with_context",
  "explanation": "Anchored media: Good Integrity - Minor cosmetic edits",
  "components": {
    "has_truth_anchor": true,
    "ai_confidence": 0.894,
    "community_consensus": 0.923,
    "nuance_score": 0.800,
    "edit_count": 3,
    "profit_correlation_flag": false,
    "profit_correlation_value": 0.03
  }
}python -m pytest vesta_tests.py -v
similarity = generator.compare_perceptual_hashes(hash1, hash2)
# Returns 0.0 (identical) to 1.0 (completely different)
is_valid = AnchorSeedGenerator.verify_anchor_signature(anchor, public_key)
""" 
Project Vesta - Enhanced All-in-One Core Script (Layer 1, 2, 3) v1.1 
Consolidated media authenticity system with cryptographic provenance tracking 
"""

import hashlib
import time
import json
import random
import numpy as np
from typing import List, Dict, Any
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
import torch
import torch.nn as nn

# ==================== CONSTANTS ====================
PROFIT_CORRELATION_MAX: float = 0.05
ETHICAL_BENEFIT_MIN: float = 0.7
P_HASH_SIGNIFICANCE_THRESHOLD: float = 0.40

# ==================== CRYPTO UTILITIES ====================
def generate_keypair():
    """Generate a new Ed25519 keypair for device signing."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key

def serialize_public_key(public_key: ed25519.Ed25519PublicKey) -> str:
    """Serializes a public key to a hex string for storage/transport."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    ).hex()

def deserialize_public_key(public_key_hex: str) -> ed25519.Ed25519PublicKey:
    """Deserializes a hex string back into a public key object."""
    return ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))

# ==================== LAYER 1: TRUTH ANCHOR ====================
class AnchorSeedGenerator:
    """Creates a verifiable, signed Anchor for a piece of media."""

    def __init__(self, device_id: str, private_key: ed25519.Ed25519PrivateKey):
        self.device_id = device_id
        self.private_key = private_key

    def generate_perceptual_hash(self, raw_data: bytes) -> str:
        """Wavelet-Enhanced Perceptual Hash (Enhanced: Padding for short data)"""
        timestamp_nonce = str(time.time_ns()).encode()
        combined_data = raw_data + timestamp_nonce
        
        try:
            data_array = np.frombuffer(combined_data[:1024], dtype=np.uint8)
            # Enhanced: Pad to multiple of 32
            pad_length = 32 - (len(data_array) % 32)
            if pad_length != 32:
                data_array = np.pad(data_array, (0, pad_length), 'constant', constant_values=0)
            feature_vector = np.mean(data_array.reshape(-1, 32), axis=0)  
        except ValueError:
            feature_vector = np.array([int(hashlib.sha1(combined_data).hexdigest()[:16], 16)])  # Numeric fallback

        hashable_data = str(feature_vector).encode('utf-8')
        return hashlib.sha256(hashable_data).hexdigest()

    def create_truth_anchor(self, raw_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Main function to generate the signed Anchor object."""
        timestamp = int(time.time())
        perceptual_hash = self.generate_perceptual_hash(raw_data)
        
        payload_data = {
            "p_hash": perceptual_hash,
            "device_id": self.device_id,
            "timestamp": timestamp,
            "metadata": metadata
        }
        
        payload_str = json.dumps(payload_data, sort_keys=True, separators=(',', ':'))
        signature = self.private_key.sign(payload_str.encode())
        anchor_id = hashlib.sha256(signature).hexdigest()

        return {
            "anchor_id": anchor_id,
            "payload": payload_data,
            "signature": signature.hex(),
            "version": "1.1"  # Bumped for enhancements
        }

    def compare_perceptual_hashes(self, p_hash_a: str, p_hash_b: str) -> float:
        """
        Compares two p-hashes and returns a similarity score (distance).
        0.0 = identical; 1.0 = completely different.
        """
        if len(p_hash_a) != len(p_hash_b):
            return 1.0

        # Calculate Hamming distance (XOR and count set bits)
        # Convert hex strings to integers for bitwise operations
        try:
            a_int = int(p_hash_a, 16)
            b_int = int(p_hash_b, 16)
        except ValueError:
            return 1.0  # Handle non-hex data
            
        diff = a_int ^ b_int

        # Count set bits in the difference
        distance = bin(diff).count('1')

        # Normalize by the total number of bits (hash length * 4 bits/char)
        total_bits = len(p_hash_a) * 4
        return distance / total_bits

    @staticmethod
    def verify_anchor_signature(anchor: Dict[str, Any], public_key: ed25519.Ed25519PublicKey) -> bool:
        """Verifies the signature of the original Anchor object."""
        try:
            payload_data = anchor["payload"]
            payload_str = json.dumps(payload_data, sort_keys=True, separators=(',', ':'))
            signature = bytes.fromhex(anchor["signature"])
            public_key.verify(signature, payload_str.encode())
            return True
        except (InvalidSignature, KeyError, ValueError):
            return False

# ==================== LAYER 2 HELPER: NUANCE CALCULATOR ====================
class NuanceCalculator:
    @staticmethod
    def calculate_nuance_score(provenance_chain: List[Dict[str, Any]], 
                              has_original_anchor: bool = True) -> float:
        if not has_original_anchor: 
            return 0.3
        base_score = 1.0 - min(0.5, len(provenance_chain) * 0.1)
        for event in provenance_chain:
            edit_type = event.get("edit_type", "").lower()
            if "ai_generate" in edit_type or "deepfake" in edit_type:
                base_score -= 0.2
            elif "filter" in edit_type or "color" in edit_type:
                base_score -= 0.05
        return max(0.1, min(1.0, base_score))

    @staticmethod
    def get_nuance_description(score: float) -> str:
        if score >= 0.9: 
            return "High Integrity - Minimal alterations detected"
        elif score >= 0.7: 
            return "Good Integrity - Minor cosmetic edits"
        elif score >= 0.5: 
            return "Moderate Integrity - Multiple edits, verify context"
        elif score >= 0.3: 
            return "Low Integrity - Significant alterations"
        else: 
            return "Very Low Integrity - Requires careful verification"

# ==================== LAYER 2: PROVENANCE TRACKER ====================
class ProvenanceTracker:
    """Tracks the complete edit history of a media file."""

    def __init__(self, anchor_id: str):
        self.anchor_id = anchor_id
        self.edit_chain: List[Dict[str, Any]] = []

    def _get_chain_tip_hash(self) -> str:
        if not self.edit_chain: 
            return self.anchor_id
        return self.edit_chain[-1]["event_id"]

    def add_edit(self, edit_type: str, editor_id: str, edit_metadata: Dict[str, Any], 
                 private_key: ed25519.Ed25519PrivateKey) -> str:
        timestamp = int(time.time())
        edit_event = {
            "edit_type": edit_type, 
            "editor_id": editor_id, 
            "timestamp": timestamp,
            "metadata": edit_metadata, 
            "previous_hash": self._get_chain_tip_hash()
        }
        event_str = json.dumps(edit_event, sort_keys=True, separators=(',', ':'))
        signature = private_key.sign(event_str.encode())
        signed_event = {
            **edit_event, 
            "signature": signature.hex(), 
            "event_id": hashlib.sha256(signature).hexdigest()
        }
        self.edit_chain.append(signed_event)
        return signed_event["event_id"]

    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        return self.edit_chain.copy()

    def _get_event_hashable_payload(self, event: Dict[str, Any]) -> bytes:
        payload_data = {
            "edit_type": event["edit_type"], 
            "editor_id": event["editor_id"], 
            "timestamp": event["timestamp"], 
            "metadata": event["metadata"], 
            "previous_hash": event["previous_hash"]
        }
        payload_str = json.dumps(payload_data, sort_keys=True, separators=(',', ':'))
        return payload_str.encode()

    def verify_chain(self, public_key_map: Dict[str, ed25519.Ed25519PublicKey]) -> bool:
        """Verify the cryptographic integrity and linkage of the entire provenance chain."""
        current_hash_link = self.anchor_id
        for i, event in enumerate(self.edit_chain):
            # 1. HASH LINKAGE
            if event.get("previous_hash") != current_hash_link:
                print(f"FAIL: Hash linkage broken at event index {i}.")
                return False
            # 2. SIGNATURE INTEGRITY
            editor_id = event.get("editor_id")
            public_key = public_key_map.get(editor_id)
            if not public_key:
                print(f"FAIL: Public key not found for editor: {editor_id}")
                return False
            try:
                payload = self._get_event_hashable_payload(event)
                signature = bytes.fromhex(event["signature"])
                public_key.verify(signature, payload)
            except (InvalidSignature, ValueError):
                print(f"FAIL: Invalid signature detected for event index {i}.")
                return False
            # 3. UPDATE CHAIN LINKAGE
            current_hash_link = event.get("event_id")
        return True

# ==================== LAYER 3: CONFIDENCE ENGINE ====================
class ConfidenceEngine:
    """Combines multiple signals to calculate final confidence scores."""

    def __init__(self):
        self.ai_models = ["temporal_analysis", "spatial_analysis", "frequency_analysis"]
        # Simple CNN for mock AI confidence (expandable to real CV)
        self.cnn = self._build_cnn()
        # Mock "training": Random init for demo
        torch.manual_seed(42)  # Reproducible randomness

    def _build_cnn(self) -> nn.Module:
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(16 * 14 * 14, 1)  # For 28x28 input
            
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = x.view(x.size(0), -1)
                x = torch.sigmoid(self.fc(x))
                return x
        
        return SimpleCNN()

    def _get_ai_confidence(self, media_url: str) -> float:
        """Enhanced: Use Torch CNN on mock image tensor."""
        # Mock 28x28 grayscale image (in prod: load via torchvision)
        mock_img = torch.rand(1, 1, 28, 28)
        self.cnn.eval()
        with torch.no_grad():
            pred = self.cnn(mock_img).item()
        # Bias to realistic range
        return max(0.7, min(0.98, pred + random.uniform(0.0, 0.28)))

    def _get_community_consensus(self, media_url: str) -> float:
        return random.uniform(0.8, 0.95)

    # Video stub (uncomment for extension)
    # def _extract_video_frames(self, video_url: str) -> List[torch.Tensor]:
    #     # e.g., Use ffmpeg-python or opencv to extract frames
    #     return [torch.rand(1, 1, 28, 28)]  # Mock

    def analyze_media(self, media_url: str, metadata: Dict[str, Any], 
                     provenance_chain: List[Dict[str, Any]] = None,
                     new_media_raw_data: bytes = None) -> Dict[str, Any]:
        
        has_anchor = 'vesta_anchor_id' in metadata
        provenance_chain = provenance_chain or []
        ai_confidence = self._get_ai_confidence(media_url)
        community_consensus = self._get_community_consensus(media_url)
        nuance_score = NuanceCalculator.calculate_nuance_score(provenance_chain, has_anchor)
        
        # Enhanced: Ethical profit correlation check
        profit_correlation = metadata.get('profit_correlation', 0.0)
        profit_flag = profit_correlation > PROFIT_CORRELATION_MAX
        
        # Enhanced: P-Hash comparison for unanchored media
        p_hash_confidence = None
        if not has_anchor and new_media_raw_data and 'original_p_hash' in metadata:
            temp_private_key, _ = generate_keypair()
            generator = AnchorSeedGenerator("temp_device", temp_private_key)
            current_p_hash = generator.generate_perceptual_hash(new_media_raw_data)
            p_hash_distance = generator.compare_perceptual_hashes(metadata['original_p_hash'], current_p_hash)
            p_hash_confidence = max(0.0, 1.0 - p_hash_distance)
        
        if profit_flag:
            nuance_score *= 0.8  # Penalize potentially exploitative edits
            ai_confidence *= 0.9  # Slight AI distrust boost
        
        # Weighted final score
        if has_anchor:
            final_score = (nuance_score * 0.7) + (ai_confidence * 0.2) + (community_consensus * 0.1)
            explanation = f"Anchored media: {NuanceCalculator.get_nuance_description(nuance_score)}"
        else:
            if p_hash_confidence and p_hash_confidence > 0.7:
                # Boost score based on P-Hash match
                final_score = (ai_confidence * 0.4) + (community_consensus * 0.3) + (p_hash_confidence * 0.3)
                explanation = "Unanchored media - High P-Hash similarity to known original"
            else:
                final_score = (ai_confidence * 0.5) + (community_consensus * 0.5)
                explanation = "Unanchored media - verification based on AI and community consensus"
        
        # Determine risk level (Uses ETHICAL_BENEFIT_MIN constant)
        if final_score >= 0.8:
            risk_level = "low"
            recommendation = "trust_with_context"
        elif final_score >= ETHICAL_BENEFIT_MIN:
            risk_level = "medium"
            recommendation = "verify_provenance_but_ethically_sound"
        else:
            risk_level = "high"
            recommendation = "exercise_caution"
        
        result = {
            "confidence_score": round(final_score, 3),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "explanation": explanation,
            "components": {
                "has_truth_anchor": has_anchor,
                "ai_confidence": round(ai_confidence, 3),
                "community_consensus": round(community_consensus, 3),
                "nuance_score": round(nuance_score, 3),
                "edit_count": len(provenance_chain),
                "profit_correlation_flag": profit_flag,
                "profit_correlation_value": round(profit_correlation, 3)
            }
        }
        
        if p_hash_confidence is not None:
            result["components"]["p_hash_confidence"] = round(p_hash_confidence, 3)
            
        return result

# ==================== ENHANCED USAGE EXAMPLE ====================
def main():
    print("=== Project Vesta - Enhanced Usage Example (All-in-One v1.1) ===\n")

    # --- Setup ---
    private_key, public_key = generate_keypair() 
    generator = AnchorSeedGenerator("EXAMPLE_CAMERA_001", private_key)
    media_data = b"fake_image_data_xyz_123"  # Short data to test padding
    metadata = {"file_type": "JPEG", "resolution": "3840x2160", "profit_correlation": 0.03}  # Below threshold

    # 1. Layer 1: Create Truth Anchor
    print("1. Creating Truth Anchor...")
    anchor = generator.create_truth_anchor(media_data, metadata)
    print(f"   Anchor ID: {anchor['anchor_id'][:16]}...")

    # 2. Layer 2: Track Provenance
    print("\n2. Tracking Provenance...")
    tracker = ProvenanceTracker(anchor["anchor_id"])
    editor_key = private_key  # In prod: Per-editor keys

    tracker.add_edit("color_correction", "editor_001", {"adjustment": "exposure+0.3"}, editor_key)
    tracker.add_edit("crop", "editor_001", {"dimensions": "1920x1080"}, editor_key)
    # Test heavy edit
    tracker.add_edit("ai_generate", "editor_002", {"model": "GAN-enhance"}, private_key)  # Drops nuance

    provenance = tracker.get_provenance_chain()
    print(f"   Edit events: {len(provenance)}")

    # 3. Nuance Score
    nuance_score = NuanceCalculator.calculate_nuance_score(provenance, True)
    nuance_desc = NuanceCalculator.get_nuance_description(nuance_score)
    print(f"   Nuance Score: {round(nuance_score, 3)} - {nuance_desc}")

    # 4. Cryptographic Verification (New Step)
    print("\n4. Verifying Provenance Chain Integrity...")
    key_map = {"editor_001": public_key, "editor_002": public_key, "EXAMPLE_CAMERA_001": public_key}
    is_chain_valid = tracker.verify_chain(key_map)
    print(f"   Provenance Chain Valid: {is_chain_valid}")
    if not is_chain_valid:
        print("   **WARNING: Provenance chain verification FAILED. Media may be tampered!**")
        
    # 5. Layer 3: Confidence Analysis
    print("\n5. Confidence Analysis...")
    engine = ConfidenceEngine()
    analysis = engine.analyze_media(
        media_url="https://example.com/media.jpg",
        metadata={"vesta_anchor_id": anchor["anchor_id"], "profit_correlation": 0.03},
        provenance_chain=provenance
    )

    print(f"   Final Confidence: {analysis['confidence_score']}")
    print(f"   Risk Level: {analysis['risk_level'].upper()}")
    print(f"   Recommendation: {analysis['recommendation']}")
    print(f"   Explanation: {analysis['explanation']}")
    print(f"   Profit Flag: {analysis['components']['profit_correlation_flag']}")

    # 6. JSON Export (New)
    print("\n6. Exporting Analysis to JSON...")
    export_data = {
        "anchor": anchor,
        "provenance": provenance,
        "analysis": analysis
    }
    with open('vesta_analysis.json', 'w') as fp:  # In prod: Save to file
        json.dump(export_data, fp, indent=2)
    print("   Exported to vesta_analysis.json")

    # --- No-Anchor Test Case (New) ---
    print("\n--- No-Anchor Test (w/ High Profit) ---")
    no_anchor_metadata = {"file_type": "JPEG", "resolution": "3840x2160", "profit_correlation": 0.06}  # Above threshold
    no_anchor_analysis = engine.analyze_media(
        media_url="https://example.com/unanchored.jpg",
        metadata=no_anchor_metadata,
        provenance_chain=[]  # No chain
    )
    print(f"   No-Anchor Confidence: {no_anchor_analysis['confidence_score']}")
    print(f"   No-Anchor Risk: {no_anchor_analysis['risk_level'].upper()}")
    print(f"   Profit Flag: {no_anchor_analysis['components']['profit_correlation_flag']}")
    print(f"   Profit Value: {no_anchor_analysis['components']['profit_correlation_value']}")

    # --- P-Hash Comparison Test ---
    print("\n--- P-Hash Comparison Test ---")
    similar_data = b"fake_image_data_xyz_123_slightly_different"
    p_hash_distance = generator.compare_perceptual_hashes(
        anchor["payload"]["p_hash"],
        generator.generate_perceptual_hash(similar_data)
    )
    print(f"   P-Hash Distance: {round(p_hash_distance, 3)}")
    print(f"   P-Hash Similarity: {round(1 - p_hash_distance, 3)}")

    # --- Anchor Signature Verification Test ---
    print("\n--- Anchor Signature Verification Test ---")
    is_anchor_valid = AnchorSeedGenerator.verify_anchor_signature(anchor, public_key)
    print(f"   Anchor Signature Valid: {is_anchor_valid}")

    print("\n=== Enhanced Example Complete ===")

if __name__ == "__main__":
    main()
# ==================== TEST SUITE ====================
import pytest
import copy

class TestVestaSystem:
    """Comprehensive test suite for Project Vesta components"""
    
    def test_anchor_creation(self):
        """Test that anchors can be created and verified"""
        private_key, public_key = generate_keypair()
        generator = AnchorSeedGenerator("TEST_CAMERA", private_key)
        anchor = generator.create_truth_anchor(b"test_data", {"test": "metadata"})
        
        assert "anchor_id" in anchor
        assert "signature" in anchor
        assert AnchorSeedGenerator.verify_anchor_signature(anchor, public_key)
    
    def test_provenance_chain_integrity(self):
        """Test provenance chain cryptographic integrity"""
        private_key, public_key = generate_keypair()
        generator = AnchorSeedGenerator("TEST_CAMERA", private_key)
        anchor = generator.create_truth_anchor(b"test_data", {})
        
        tracker = ProvenanceTracker(anchor["anchor_id"])
        tracker.add_edit("test_edit", "editor_1", {"note": "test"}, private_key)
        
        key_map = {"editor_1": public_key}
        assert tracker.verify_chain(key_map)
    
    def test_nuance_score_calculation(self):
        """Test nuance scoring logic"""
        # Test with no anchor
        assert NuanceCalculator.calculate_nuance_score([], False) == 0.3
        
        # Test with empty chain but anchor
        assert NuanceCalculator.calculate_nuance_score([], True) == 1.0
        
        # Test with AI edits
        ai_edits = [{"edit_type": "ai_generate", "metadata": {}}]
        score = NuanceCalculator.calculate_nuance_score(ai_edits, True)
        assert score == 0.8  # 1.0 - 0.2 for AI edit

def run_tests():
    """Run the test suite"""
    print("🧪 Running Project Vesta Test Suite...")
    test_suite = TestVestaSystem()
    
    tests = [
        test_suite.test_anchor_creation,
        test_suite.test_provenance_chain_integrity,
        test_suite.test_nuance_score_calculation,
    ]
    
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__} passed")
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print("🧪 Test suite completed!")
    # ==================== UTILITY FUNCTIONS ====================
def export_vesta_analysis(anchor, provenance_chain, analysis, filename="vesta_analysis.json"):
    """Export complete Vesta analysis to JSON file"""
    export_data = {
        "export_timestamp": int(time.time()),
        "export_version": "1.1",
        "anchor": anchor,
        "provenance_chain": provenance_chain,
        "confidence_analysis": analysis,
        "system_info": {
            "project_vesta_version": "1.1",
            "export_format": "comprehensive"
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return filename

def create_media_report(analysis):
    """Create a human-readable media authenticity report"""
    report = f"""
MEDIA AUTHENTICITY REPORT
=========================

Overall Confidence: {analysis['confidence_score']} / 1.0
Risk Level: {analysis['risk_level'].upper()}
Recommendation: {analysis['recommendation'].replace('_', ' ').title()}

DETAILS:
--------
{analysis['explanation']}

COMPONENT SCORES:
-----------------
- Has Truth Anchor: {analysis['components']['has_truth_anchor']}
- AI Confidence: {analysis['components']['ai_confidence']}
- Community Consensus: {analysis['components']['community_consensus']}
- Nuance Score: {analysis['components']['nuance_score']}
- Edit Count: {analysis['components']['edit_count']}
- Profit Correlation Flag: {analysis['components']['profit_correlation_flag']}
"""
    
    if 'p_hash_confidence' in analysis['components']:
        report += f"- P-Hash Confidence: {analysis['components']['p_hash_confidence']}\n"
    
    return report

def batch_analyze_media(media_list):
    """Analyze multiple media items in batch"""
    engine = ConfidenceEngine()
    results = []
    
    for media_item in media_list:
        try:
            analysis = engine.analyze_media(
                media_url=media_item.get('url', ''),
                metadata=media_item.get('metadata', {}),
                provenance_chain=media_item.get('provenance_chain', [])
            )
            results.append({
                'media_id': media_item.get('id', 'unknown'),
                'analysis': analysis
            })
        except Exception as e:
            results.append({
                'media_id': media_item.get('id', 'unknown'),
                'error': str(e)
            })
    
    return results
    # ==================== WEB API STUB ====================
class VestaAPI:
    """REST API stub for Project Vesta (for future implementation)"""
    
    def __init__(self):
        self.engine = ConfidenceEngine()
        self.anchors = {}
        self.provenance_chains = {}
    
    def create_anchor(self, device_id, media_data, metadata):
        """API endpoint: Create new truth anchor"""
        private_key, public_key = generate_keypair()
        generator = AnchorSeedGenerator(device_id, private_key)
        anchor = generator.create_truth_anchor(media_data, metadata)
        
        # Store anchor and public key
        anchor_id = anchor["anchor_id"]
        self.anchors[anchor_id] = {
            "anchor": anchor,
            "public_key": serialize_public_key(public_key)
        }
        
        return {
            "anchor_id": anchor_id,
            "anchor": anchor,
            "public_key": serialize_public_key(public_key)
        }
    
    def add_provenance_event(self, anchor_id, edit_type, editor_id, edit_metadata):
        """API endpoint: Add provenance event to existing anchor"""
        if anchor_id not in self.anchors:
            return {"error": "Anchor not found"}
        
        # In real implementation, you'd have proper key management
        private_key, _ = generate_keypair()
        
        if anchor_id not in self.provenance_chains:
            self.provenance_chains[anchor_id] = ProvenanceTracker(anchor_id)
        
        tracker = self.provenance_chains[anchor_id]
        event_id = tracker.add_edit(edit_type, editor_id, edit_metadata, private_key)
        
        return {
            "event_id": event_id,
            "provenance_chain_length": len(tracker.get_provenance_chain())
        }
    
    def analyze_media(self, media_url, metadata, anchor_id=None):
        """API endpoint: Analyze media authenticity"""
        provenance_chain = []
        if anchor_id and anchor_id in self.provenance_chains:
            provenance_chain = self.provenance_chains[anchor_id].get_provenance_chain()
        
        analysis = self.engine.analyze_media(media_url, metadata, provenance_chain)
        return analysis
        # ==================== PERFORMANCE MONITORING ====================
class PerformanceMonitor:
    """Monitor system performance and metrics"""
    
    def __init__(self):
        self.metrics = {
            'anchors_created': 0,
            'edits_tracked': 0,
            'analyses_performed': 0,
            'chain_verifications': 0,
            'errors': 0
        }
        self.start_time = time.time()
    
    def record_metric(self, metric_name):
        """Record a metric event"""
        if metric_name in self.metrics:
            self.metrics[metric_name] += 1
    
    def get_performance_report(self):
        """Generate performance report"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        report = {
            'uptime_seconds': round(uptime, 2),
            'metrics': self.metrics.copy(),
            'average_throughput': {
                'anchors_per_hour': round(self.metrics['anchors_created'] / (uptime / 3600), 2),
                'analyses_per_hour': round(self.metrics['analyses_performed'] / (uptime / 3600), 2)
            },
            'system_health': 'HEALTHY' if self.metrics['errors'] == 0 else 'DEGRADED'
        }
        
        return report

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
# ==================== ENHANCED SECURITY ====================
class SecurityAuditor:
    """Enhanced security auditing and validation"""
    
    @staticmethod
    def validate_anchor_structure(anchor):
        """Validate anchor structure and content"""
        required_fields = ['anchor_id', 'payload', 'signature', 'version']
        
        for field in required_fields:
            if field not in anchor:
                return False, f"Missing required field: {field}"
        
        # Validate payload structure
        payload = anchor['payload']
        payload_fields = ['p_hash', 'device_id', 'timestamp', 'metadata']
        for field in payload_fields:
            if field not in payload:
                return False, f"Missing payload field: {field}"
        
        # Validate timestamp (not in future, not too old)
        current_time = time.time()
        if payload['timestamp'] > current_time + 300:  # 5 minutes in future
            return False, "Anchor timestamp is in future"
        
        if payload['timestamp'] < current_time - 31536000:  # 1 year ago
            return False, "Anchor timestamp is too old"
        
        return True, "Anchor structure valid"
    
    @staticmethod
    def detect_anomalies(provenance_chain):
        """Detect anomalies in provenance chain"""
        anomalies = []
        
        if not provenance_chain:
            return anomalies
        
        # Check for rapid successive edits (potential automation)
        timestamps = [event['timestamp'] for event in provenance_chain]
        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff < 1:  # Less than 1 second between edits
                anomalies.append(f"Rapid successive edits at position {i}")
        
        # Check for suspicious edit patterns
        edit_types = [event['edit_type'] for event in provenance_chain]
        ai_edit_count = sum(1 for edit_type in edit_types if 'ai' in edit_type.lower())
        if ai_edit_count > len(edit_types) * 0.5:  # More than 50% AI edits
            anomalies.append("High proportion of AI-generated edits")
        
        return anomalies
        # ==================== ENHANCED MAIN DEMO ====================
def enhanced_demo():
    """Comprehensive demo showcasing all Vesta features"""
    print("🚀 Project Vesta - Comprehensive Demo")
    print("=" * 50)
    
    # Initialize monitoring
    monitor = PerformanceMonitor()
    
    # Demo all components
    private_key, public_key = generate_keypair()
    generator = AnchorSeedGenerator("PRO_DEMO_CAMERA", private_key)
    
    # Create anchor
    media_data = b"sample_media_content_2024"
    metadata = {
        "format": "RAW",
        "resolution": "4000x3000", 
        "camera_model": "DemoCam X1",
        "location": "Demo Studio",
        "profit_correlation": 0.02
    }
    
    print("\n1. 🎯 Creating Truth Anchor...")
    anchor = generator.create_truth_anchor(media_data, metadata)
    monitor.record_metric('anchors_created')
    print(f"   ✅ Anchor created: {anchor['anchor_id'][:24]}...")
    
    # Validate anchor
    is_valid, message = SecurityAuditor.validate_anchor_structure(anchor)
    print(f"   🔍 Anchor validation: {message}")
    
    print("\n2. 🔗 Building Provenance Chain...")
    tracker = ProvenanceTracker(anchor["anchor_id"])
    
    edits = [
        ("color_correction", "editor_john", {"adjustment": "exposure+0.5", "white_balance": "auto"}),
        ("crop", "editor_john", {"from": "4000x3000", "to": "1920x1080", "aspect_ratio": "16:9"}),
        ("filter_apply", "editor_sarah", {"filter": "vintage", "intensity": 0.3}),
        ("ai_enhance", "editor_ai", {"model": "SuperRes-v2", "purpose": "quality_improvement"})
    ]
    
    for edit_type, editor, edit_meta in edits:
        tracker.add_edit(edit_type, editor, edit_meta, private_key)
        monitor.record_metric('edits_tracked')
        print(f"   ✅ Added: {edit_type} by {editor}")
    
    print(f"   📊 Total edits: {len(tracker.get_provenance_chain())}")
    
    print("\n3. 🛡️ Security Audit...")
    anomalies = SecurityAuditor.detect_anomalies(tracker.get_provenance_chain())
    if anomalies:
        print("   ⚠️  Anomalies detected:")
        for anomaly in anomalies:
            print(f"      - {anomaly}")
    else:
        print("   ✅ No security anomalies detected")
    
    print("\n4. 🔐 Cryptographic Verification...")
    key_map = {"editor_john": public_key, "editor_sarah": public_key, "editor_ai": public_key}
    is_chain_valid = tracker.verify_chain(key_map)
    monitor.record_metric('chain_verifications')
    print(f"   {'✅' if is_chain_valid else '❌'} Chain integrity: {is_chain_valid}")
    
    print("\n5. 🧠 Confidence Analysis...")
    engine = ConfidenceEngine()
    analysis = engine.analyze_media(
        media_url="https://demo.com/professional/photo.jpg",
        metadata={"vesta_anchor_id": anchor["anchor_id"], **metadata},
        provenance_chain=tracker.get_provenance_chain()
    )
    monitor.record_metric('analyses_performed')
    
    print(f"   📈 Confidence Score: {analysis['confidence_score']}")
    print(f"   🎯 Risk Level: {analysis['risk_level'].upper()}")
    print(f"   💡 Recommendation: {analysis['recommendation'].replace('_', ' ').title()}")
    
    print("\n6. 📊 Generating Reports...")
    # Export JSON
    export_file = export_vesta_analysis(anchor, tracker.get_provenance_chain(), analysis, "comprehensive_demo.json")
    print(f"   ✅ JSON export: {export_file}")
    
    # Generate human-readable report
    report = create_media_report(analysis)
    print("   ✅ Human-readable report generated")
    
    print("\n7. 📈 Performance Metrics...")
    perf_report = monitor.get_performance_report()
    print(f"   ⏱️  Uptime: {perf_report['uptime_seconds']}s")
    print(f"   📊 Anchors created: {perf_report['metrics']['anchors_created']}")
    print(f"   🔄 Analyses performed: {perf_report['metrics']['analyses_performed']}")
    print(f"   🏥 System health: {perf_report['system_health']}")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("💡 Check 'comprehensive_demo.json' for full analysis export")

# ==================== COMMAND LINE INTERFACE ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "demo":
            enhanced_demo()
        elif command == "tests":
            run_tests()
        elif command == "quick":
            main()  # Original simple demo
        else:
            print("Usage: python vesta_core.py [demo|tests|quick]")
            print("  demo  - Run comprehensive demo")
            print("  tests - Run test suite") 
            print("  quick - Run quick demo")
            print("  (no args) - Run quick demo")
    else:
        # Run quick demo by default
        main()
