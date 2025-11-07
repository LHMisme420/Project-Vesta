"""
Project Vesta - Layer 2: Provenance Tracker
Purpose: Tracks and verifies the edit history of media files
"""

import hashlib
import time
import json
from typing import List, Dict, Any
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature
# Note: The original file had a placeholder for verify_chain. This is the complete version.

class ProvenanceTracker:
    """Tracks the complete edit history of a media file."""
    
    def __init__(self, anchor_id: str):
        self.anchor_id = anchor_id
        self.edit_chain: List[Dict[str, Any]] = []
        
    def add_edit(self, edit_type: str, editor_id: str, edit_metadata: Dict[str, Any], 
                 private_key: ed25519.Ed25519PrivateKey) -> str:
        """
        Add a new edit event to the provenance chain.
        """
        timestamp = int(time.time())
        
        # Create edit event payload (WHAT IS SIGNED)
        edit_event = {
            "edit_type": edit_type,
            "editor_id": editor_id,
            "timestamp": timestamp,
            "metadata": edit_metadata,
            "previous_hash": self._get_chain_tip_hash()
        }
        
        # Sign the edit event payload
        event_str = json.dumps(edit_event, sort_keys=True, separators=(',', ':'))
        signature = private_key.sign(event_str.encode())
        
        # Create complete signed event (WHAT IS SAVED)
        signed_event = {
            **edit_event,
            "signature": signature.hex(),
            "event_id": hashlib.sha256(signature).hexdigest()
        }
        
        self.edit_chain.append(signed_event)
        return signed_event["event_id"]
    
    def _get_chain_tip_hash(self) -> str:
        """Get hash of the most recent event in the chain."""
        if not self.edit_chain:
            return self.anchor_id
        return self.edit_chain[-1]["event_id"]
    
    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Get the complete provenance chain."""
        return self.edit_chain.copy()

    def _get_event_hashable_payload(self, event: Dict[str, Any]) -> bytes:
        """
        Recreates the canonical, hashable payload string for signing verification.
        """
        # Ensure we use ONLY the data that was signed
        payload_data = {
            "edit_type": event["edit_type"],
            "editor_id": event["editor_id"],
            "timestamp": event["timestamp"],
            "metadata": event["metadata"],
            "previous_hash": event["previous_hash"]
        }
        # Recreate the canonical JSON string used for the original signing
        payload_str = json.dumps(payload_data, sort_keys=True, separators=(',', ':'))
        return payload_str.encode()

    def verify_chain(self, public_key_map: Dict[str, ed25519.Ed25519PublicKey]) -> bool:
        """
        Verify the cryptographic integrity and linkage of the entire provenance chain.

        Args:
            public_key_map: A dictionary mapping 'editor_id' to its Public Key object.

        Returns:
            Boolean indicating if chain verification succeeded (True = Tamper-Proof).
        """
        current_hash_link = self.anchor_id  # Start with the original anchor ID

        for i, event in enumerate(self.edit_chain):
            # 1. VERIFY HASH LINKAGE (CHAINING)
            if event.get("previous_hash") != current_hash_link:
                print(f"FAIL: Hash linkage broken at event index {i}. Expected: {current_hash_link[:8]}...")
                return False

            # 2. VERIFY SIGNATURE INTEGRITY
            editor_id = event.get("editor_id")
            public_key = public_key_map.get(editor_id)

            if not public_key:
                print(f"FAIL: Public key not found for editor: {editor_id}")
                return False

            try:
                payload = self._get_event_hashable_payload(event)
                signature = bytes.fromhex(event["signature"])
                public_key.verify(signature, payload)
                
            except (InvalidSignature, ValueError) as e:
                print(f"FAIL: Invalid signature detected for event index {i}. Error: {e}")
                return False
            
            # 3. UPDATE CHAIN LINKAGE (SUCCESS)
            current_hash_link = event.get("event_id")

        return True
# vesta/constants.py

# =======================================================
# Project Vesta - Global Configuration Constants
# =======================================================

# --- Layer 3: Immune System / Audit Configuration ---

# The maximum acceptable correlation (from -1.0 to 1.0) 
# between an editor's actions and an external financial profit signal.
# A low value enforces "Zero-Drift" to combat financially-motivated tampering.
PROFIT_CORRELATION_MAX: float = 0.05 

# The minimum required system stability or ethical benefit score (from 0.0 to 1.0) 
# achieved by a proposed edit or reversal. Used by an Auditor to prioritize 
# changes that make the overall Vesta network more robust and anti-fragile.
ETHICAL_BENEFIT_MIN: float = 0.7

# A threshold defining what magnitude of perceptual hash change (Layer 1) 
# is considered a 'significant' alteration.
P_HASH_SIGNIFICANCE_THRESHOLD: float = 0.40
"""
Basic usage example for Project Vesta
"""

from vesta.layer1_anchor import AnchorSeedGenerator, crypto_utils
from vesta.layer2_provenance import ProvenanceTracker, NuanceCalculator
from vesta.layer3_immune_system import ConfidenceEngine
# Note: No need to import vesta/constants here, as the Auditor/Engine would use them internally.


def main():
    print("=== Project Vesta - Basic Usage Example ===\n")
    
    # Layer 1: Create Truth Anchor
    print("1. Creating Truth Anchor...")
    private_key, public_key = crypto_utils.generate_keypair()
    generator = AnchorSeedGenerator("EXAMPLE_CAMERA_001", private_key)
    
    # Simulate media data
    media_data = b"fake_image_data_xyz_123"
    metadata = {
        "file_type": "JPEG", 
        "resolution": "3840x2160",
        "camera_model": "Example Pro"
    }
    
    anchor = generator.create_truth_anchor(media_data, metadata)
    print(f"   Anchor ID: {anchor['anchor_id'][:16]}...")
    print(f"   Timestamp: {anchor['payload']['timestamp']}")
    
    # Layer 2: Track Provenance
    print("\n2. Tracking Provenance...")
    tracker = ProvenanceTracker(anchor["anchor_id"])
    
    # Add some edits
    editor_key = private_key # Use the same key for this example
    
    tracker.add_edit("color_correction", "editor_001", 
                    {"adjustment": "exposure+0.3", "white_balance": "auto"}, 
                    editor_key)
    
    tracker.add_edit("crop", "editor_001", 
                    {"dimensions": "1920x1080", "aspect_ratio": "16:9"}, 
                    editor_key)
    
    provenance = tracker.get_provenance_chain()
    print(f"   Edit events: {len(provenance)}")
    
    # Calculate nuance score (remains Step 2 logic)
    nuance_score = NuanceCalculator.calculate_nuance_score(provenance, True)
    nuance_desc = NuanceCalculator.get_nuance_description(nuance_score)
    print(f"   Nuance Score: {nuance_score} - {nuance_desc}")
    
    # --- NEW STEP 4: Cryptographic Verification ---
    print("\n4. Verifying Provenance Chain Integrity...")

    # Map of entity IDs to their public keys (needed for verification)
    key_map = {
        "editor_001": public_key,
        "EXAMPLE_CAMERA_001": public_key 
    }

    is_chain_valid = tracker.verify_chain(key_map)

    print(f"   Provenance Chain Valid: {is_chain_valid}")
    if not is_chain_valid:
        print("   **WARNING: Provenance chain verification FAILED. Media may be tampered!**")
        
    # Layer 3: Confidence Analysis (Now Step 5)
    print("\n5. Confidence Analysis...")
    engine = ConfidenceEngine()
    
    analysis = engine.analyze_media(
        media_url="https://example.com/media.jpg",
        metadata={"vesta_anchor_id": anchor["anchor_id"]},
        provenance_chain=provenance
    )
    
    print(f"   Final Confidence: {analysis['confidence_score']}")
    print(f"   Risk Level: {analysis['risk_level'].upper()}")
    print(f"   Recommendation: {analysis['recommendation']}")
    print(f"   Explanation: {analysis['explanation']}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
    # =================================================================
# FILE: vesta/constants.py (New File)
# =================================================================
# Project Vesta - Global Configuration Constants

PROFIT_CORRELATION_MAX: float = 0.05 
ETHICAL_BENEFIT_MIN: float = 0.7
P_HASH_SIGNIFICANCE_THRESHOLD: float = 0.40


# =================================================================
# FILE: vesta/crypto_utils.py
# =================================================================
import hashlib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def generate_keypair():
    """Generate a new Ed25519 keypair for device signing."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key

def private_key_to_bytes(private_key: ed25519.Ed25519PrivateKey) -> bytes:
    """Serialize private key to bytes."""
    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )

def bytes_to_private_key(key_bytes: bytes) -> ed25519.Ed25519PrivateKey:
    """Load private key from bytes."""
    return ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)


# =================================================================
# FILE: vesta/layer1_anchor.py
# =================================================================
import time
import json
import numpy as np
from typing import Dict, Any
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import hashes # Required for original design, though not used in final impl

class AnchorSeedGenerator:
    
    def __init__(self, device_id: str, private_key: ed25519.Ed25519PrivateKey):
        self.device_id = device_id
        self.private_key = private_key
        
    def generate_perceptual_hash(self, raw_data: bytes) -> str:
        """Wavelet-Enhanced Perceptual Hash (Mock Implementation)"""
        timestamp_nonce = str(time.time_ns()).encode()
        combined_data = raw_data + timestamp_nonce
        
        try:
            # Simulate W-PHash: Hash of a low-frequency feature vector
            data_array = np.frombuffer(combined_data[:1024], dtype=np.uint8)
            feature_vector = np.mean(data_array.reshape(-1, 32), axis=0)  
        except ValueError:
            feature_vector = np.array([hashlib.sha1(combined_data).hexdigest()[:16]]) 

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
            "version": "1.0"
        }

    def verify_anchor(self, anchor_data: Dict[str, Any], public_key: ed25519.Ed25519PublicKey) -> bool:
        """Verify the integrity of an anchor."""
        try:
            payload_str = json.dumps(anchor_data["payload"], sort_keys=True, separators=(',', ':'))
            signature = bytes.fromhex(anchor_data["signature"])
            public_key.verify(signature, payload_str.encode())
            return True
        except Exception:
            return False


# =================================================================
# FILE: vesta/layer2_provenance/nuance_calculator.py
# =================================================================
from typing import List

class NuanceCalculator:
    
    @staticmethod
    def calculate_nuance_score(provenance_chain: List[Dict[str, Any]], 
                              has_original_anchor: bool = True) -> float:
        """Calculate a nuanced integrity score (0.0 to 1.0)."""
        if not has_original_anchor:
            return 0.3
        
        base_score = 1.0
        num_edits = len(provenance_chain)
        
        edit_penalty = min(0.5, num_edits * 0.1)
        base_score -= edit_penalty
        
        for event in provenance_chain:
            edit_type = event.get("edit_type", "")
            if "ai_generate" in edit_type.lower() or "deepfake" in edit_type.lower():
                base_score -= 0.2
            elif "filter" in edit_type.lower() or "color" in edit_type.lower():
                base_score -= 0.05
        
        return max(0.1, min(1.0, base_score))
    
    @staticmethod
    def get_nuance_description(score: float) -> str:
        """Get human-readable description of nuance score."""
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


# =================================================================
# FILE: vesta/layer2_provenance.py (UPDATED)
# =================================================================
from cryptography.exceptions import InvalidSignature

class ProvenanceTracker:
    
    def __init__(self, anchor_id: str):
        self.anchor_id = anchor_id
        self.edit_chain: List[Dict[str, Any]] = []
        
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
    
    def _get_chain_tip_hash(self) -> str:
        if not self.edit_chain:
            return self.anchor_id
        return self.edit_chain[-1]["event_id"]
    
    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        return self.edit_chain.copy()

    def _get_event_hashable_payload(self, event: Dict[str, Any]) -> bytes:
        """Recreates the canonical, hashable payload string for signing verification."""
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
            # 1. VERIFY HASH LINKAGE
            if event.get("previous_hash") != current_hash_link:
                print(f"FAIL: Hash linkage broken at event index {i}.")
                return False

            # 2. VERIFY SIGNATURE INTEGRITY
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


# =================================================================
# FILE: vesta/layer3_immune_system/confidence_engine.py
# =================================================================
import random
from vesta.layer2_provenance.nuance_calculator import NuanceCalculator
from vesta.constants import PROFIT_CORRELATION_MAX, ETHICAL_BENEFIT_MIN

class ConfidenceEngine:
    """Combines multiple signals to calculate final confidence scores."""
    
    def __init__(self):
        self.ai_models = ["temporal_analysis", "spatial_analysis", "frequency_analysis"]
        # Example of utilizing constants (Though not used in final scoring formula below)
        # print(f"Audit Threshold: Max Profit Correlation={PROFIT_CORRELATION_MAX}") 
        
    def analyze_media(self, media_url: str, metadata: Dict[str, Any], 
                     provenance_chain: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive media analysis combining all verification signals."""
        has_anchor = 'vesta_anchor_id' in metadata
        provenance_chain = provenance_chain or []
        
        ai_confidence = self._get_ai_confidence(media_url)
        community_consensus = self._get_community_consensus(media_url)
        nuance_score = NuanceCalculator.calculate_nuance_score(provenance_chain, has_anchor)
        
        # Weighted final score
        if has_anchor:
            final_score = (nuance_score * 0.7) + (ai_confidence * 0.2) + (community_consensus * 0.1)
            explanation = f"Anchored media: {NuanceCalculator.get_nuance_description(nuance_score)}"
        else:
            final_score = (ai_confidence * 0.5) + (community_consensus * 0.5)
            explanation = "Unanchored media - verification based on AI and community consensus"
        
        # Determine risk level
        if final_score >= 0.8:
            risk_level = "low"
            recommendation = "trust_with_context"
        else:
            # Check a complex audit rule based on a constant
            if final_score >= ETHICAL_BENEFIT_MIN: # Example usage of a constant
                risk_level = "medium"
                recommendation = "verify_provenance_but_ethically_sound"
            else:
                risk_level = "high"
                recommendation = "exercise_caution"
        
        return {
            "confidence_score": round(final_score, 3),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "explanation": explanation,
            "components": {
                "has_truth_anchor": has_anchor,
                "ai_confidence": round(ai_confidence, 3),
                "community_consensus": round(community_consensus, 3),
                "nuance_score": round(nuance_score, 3),
                "edit_count": len(provenance_chain)
            }
        }
    
    def _get_ai_confidence(self, media_url: str) -> float:
        return random.uniform(0.7, 0.98)
    
    def _get_community_consensus(self, media_url: str) -> float:
        return random.uniform(0.8, 0.95)


# =================================================================
# FILE: examples/basic_usage.py (Main Runner)
# =================================================================
# Simulate imports from the modules defined above
# NOTE: This section MUST be run separately and relies on the class definitions above.

def main():
    print("=== Project Vesta - Basic Usage Example (Integrated) ===\n")
    
    # --- Setup ---
    # Assuming crypto_utils.py is available
    private_key, public_key = generate_keypair() 
    generator = AnchorSeedGenerator("EXAMPLE_CAMERA_001", private_key)
    media_data = b"fake_image_data_xyz_123"
    metadata = {"file_type": "JPEG", "resolution": "3840x2160"}
    
    # 1. Layer 1: Create Truth Anchor
    print("1. Creating Truth Anchor...")
    anchor = generator.create_truth_anchor(media_data, metadata)
    print(f"   Anchor ID: {anchor['anchor_id'][:16]}...")
    
    # 2. Layer 2: Track Provenance
    print("\n2. Tracking Provenance...")
    tracker = ProvenanceTracker(anchor["anchor_id"])
    editor_key = private_key
    
    tracker.add_edit("color_correction", "editor_001", 
                    {"adjustment": "exposure+0.3"}, editor_key)
    tracker.add_edit("crop", "editor_001", 
                    {"dimensions": "1920x1080"}, editor_key)
    
    provenance = tracker.get_provenance_chain()
    print(f"   Edit events: {len(provenance)}")
    
    # 3. Nuance Score
    nuance_score = NuanceCalculator.calculate_nuance_score(provenance, True)
    nuance_desc = NuanceCalculator.get_nuance_description(nuance_score)
    print(f"   Nuance Score: {round(nuance_score, 3)} - {nuance_desc}")
    
    # 4. Cryptographic Verification (New Step)
    print("\n4. Verifying Provenance Chain Integrity...")
    key_map = {"editor_001": public_key, "EXAMPLE_CAMERA_001": public_key}

    is_chain_valid = tracker.verify_chain(key_map)

    print(f"   Provenance Chain Valid: {is_chain_valid}")
    if not is_chain_valid:
        print("   **WARNING: Provenance chain verification FAILED. Media may be tampered!**")
        
    # 5. Layer 3: Confidence Analysis
    print("\n5. Confidence Analysis...")
    engine = ConfidenceEngine()
    
    analysis = engine.analyze_media(
        media_url="https://example.com/media.jpg",
        metadata={"vesta_anchor_id": anchor["anchor_id"]},
        provenance_chain=provenance
    )
    
    print(f"   Final Confidence: {analysis['confidence_score']}")
    print(f"   Risk Level: {analysis['risk_level'].upper()}")
    print(f"   Recommendation: {analysis['recommendation']}")
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    # You would need to structure the imports correctly in your local environment
    # to run this successfully.
    # main() 
    print("\n***NOTE: To run this code, ensure Python module structure and imports are correct.***")
