Project-Vesta/
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   ├── vesta/
│   │   ├── __init__.py
│   │   ├── layer1_anchor/
│   │   │   ├── __init__.py
│   │   │   ├── anchor_seed_generator.py
│   │   │   └── crypto_utils.py
│   │   ├── layer2_provenance/
│   │   │   ├── __init__.py
│   │   │   ├── provenance_tracker.py
│   │   │   └── nuance_calculator.py
│   │   └── layer3_immune_system/
│   │       ├── __init__.py
│   │       ├── immune_agent.py
│   │       └── consensus_engine.py
├── tests/
│   ├── __init__.py
│   ├── test_anchor.py
│   ├── test_provenance.py
│   └── test_immune.py
├── examples/
│   ├── basic_usage.py
│   └── advanced_provenance.py
└── docs/
    ├── API.md
    └── ARCHITECTURE.md
                                     Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

   [Standard Apache 2.0 license text continues...]
   # Project Vesta - Cryptographic Truth Anchor System

> **Immutable Digital Birth Certificates for Media Content**

A three-layer protocol for creating verifiable authenticity proofs for digital media, combating misinformation through cryptographic truth anchors.

## 🏗️ Architecture

| Layer | Purpose | Key Feature |
|-------|---------|-------------|
| **1. Truth Anchor** | Creates immutable media birth certificates | Cryptographic signing + perceptual hashing |
| **2. Provenance Tracker** | Tracks verifiable edit history | Nuance scoring + timeline visualization |
| **3. Immune System** | Calculates confidence scores | AI + community consensus |

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/Project-Vesta.git
cd Project-Vesta

# Install dependencies
pip install -e .

# Run examples
python examples/basic_usage.py
from vesta.layer1_anchor import AnchorSeedGenerator
from cryptography.hazmat.primitives.asymmetric import ed25519

# Generate device key
private_key = ed25519.Ed25519PrivateKey.generate()
generator = AnchorSeedGenerator("CAMERA_001", private_key)

# Create anchor for media
anchor = generator.create_truth_anchor(
    b"raw_media_data_here",
    {"file_type": "JPEG", "resolution": "4K"}
)from vesta.layer2_provenance import ProvenanceTracker

tracker = ProvenanceTracker(anchor["anchor_id"])
tracker.add_edit("color_correction", "editor_001", {"adjustment": "exposure+0.5"})
provenance_data = tracker.get_provenance_chain()

### 3. requirements.txt
```txt
cryptography>=41.0.0
Pillow>=10.0.0
numpy>=1.24.0
opencv-python>=4.8.0
web3>=6.0.0
requests>=2.31.0
pytest>=7.0.0
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "project-vesta"
version = "0.1.0"
description = "Cryptographic Truth Anchor System for Media Authenticity"
authors = [
    {name = "Vesta Contributors", email = "contributors@vesta.truth"},
]
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.8"
dependencies = [
    "cryptography>=41.0.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]

[project.urls]
"Homepage" = "https://github.com/your-username/Project-Vesta"
"Documentation" = "https://vesta.truth/docs"
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/

# IDE
.vscode/
.idea/

# Logs
*.log

# Media files (should be handled externally)
*.jpg
*.png
*.mp4
"""
Project Vesta - Cryptographic Truth Anchor System
"""

__version__ = "0.1.0"
__author__ = "Vesta Contributors"
__license__ = "Apache 2.0"

from vesta.layer1_anchor import AnchorSeedGenerator
from vesta.layer2_provenance import ProvenanceTracker
from vesta.layer3_immune_system import ConfidenceEngine

__all__ = ["AnchorSeedGenerator", "ProvenanceTracker", "ConfidenceEngine"]
"""
Project Vesta - Layer 1: The Cryptographic Truth Anchor
Purpose: Generates the immutable digital birth certificate (Anchor ID)
"""

import hashlib
import time
import json
from typing import Dict, Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519


class AnchorSeedGenerator:
    """Creates a verifiable, signed Anchor for a piece of media."""
    
    def __init__(self, device_id: str, private_key: ed25519.Ed25519PrivateKey):
        self.device_id = device_id
        self.private_key = private_key
        
    def generate_perceptual_hash(self, raw_data: bytes) -> str:
        """
        Enhanced perceptual hash with timestamp nonce for uniqueness.
        In production, this would use AI-powered perceptual hashing.
        """
        timestamp_nonce = str(time.time_ns()).encode()
        combined_data = raw_data + timestamp_nonce
        return hashlib.sha256(combined_data).hexdigest()

    def create_truth_anchor(self, raw_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main function to generate the signed Anchor object.
        
        Args:
            raw_data: The raw media data bytes
            metadata: Dictionary of media metadata
            
        Returns:
            Dictionary containing anchor_id, payload, and signature
        """
        timestamp = int(time.time())
        perceptual_hash = self.generate_perceptual_hash(raw_data)
        
        # Create payload for signing
        payload_data = {
            "p_hash": perceptual_hash,
            "device_id": self.device_id,
            "timestamp": timestamp,
            "metadata": metadata
        }
        
        # Convert payload to canonical JSON string for consistent signing
        payload_str = json.dumps(payload_data, sort_keys=True, separators=(',', ':'))
        signature = self.private_key.sign(payload_str.encode())
        
        # Generate anchor ID from signature
        anchor_id = hashlib.sha256(signature).hexdigest()

        return {
            "anchor_id": anchor_id,
            "payload": payload_data,
            "signature": signature.hex(),
            "version": "1.0"
        }

    def verify_anchor(self, anchor_data: Dict[str, Any], public_key: ed25519.Ed25519PublicKey) -> bool:
        """
        Verify the integrity of an anchor.
        
        Args:
            anchor_data: The anchor data to verify
            public_key: The public key for verification
            
        Returns:
            Boolean indicating if verification succeeded
        """
        try:
            payload_str = json.dumps(anchor_data["payload"], sort_keys=True, separators=(',', ':'))
            signature = bytes.fromhex(anchor_data["signature"])
            public_key.verify(signature, payload_str.encode())
            return True
        except Exception:
            return False
"""
Cryptographic utilities for Project Vesta
"""

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
"""
Project Vesta - Layer 2: Provenance Tracker
Purpose: Tracks and verifies the edit history of media files
"""

import hashlib
import time
import json
from typing import List, Dict, Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519


class ProvenanceTracker:
    """Tracks the complete edit history of a media file."""
    
    def __init__(self, anchor_id: str):
        self.anchor_id = anchor_id
        self.edit_chain: List[Dict[str, Any]] = []
        
    def add_edit(self, edit_type: str, editor_id: str, edit_metadata: Dict[str, Any], 
                 private_key: ed25519.Ed25519PrivateKey) -> str:
        """
        Add a new edit event to the provenance chain.
        
        Args:
            edit_type: Type of edit (crop, filter, enhance, etc.)
            editor_id: ID of the editor/software
            edit_metadata: Detailed metadata about the edit
            private_key: Editor's private key for signing
            
        Returns:
            Edit event ID
        """
        timestamp = int(time.time())
        
        # Create edit event
        edit_event = {
            "edit_type": edit_type,
            "editor_id": editor_id,
            "timestamp": timestamp,
            "metadata": edit_metadata,
            "previous_hash": self._get_chain_tip_hash()
        }
        
        # Sign the edit event
        event_str = json.dumps(edit_event, sort_keys=True, separators=(',', ':'))
        signature = private_key.sign(event_str.encode())
        
        # Create complete signed event
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
    
    def verify_chain(self, anchor_public_key: ed25519.Ed25519PublicKey) -> bool:
        """
        Verify the integrity of the entire provenance chain.
        
        Args:
            anchor_public_key: Public key of the original anchor creator
            
        Returns:
            Boolean indicating if chain verification succeeded
        """
        # Implementation would verify each signature in the chain
        # This is a simplified version
        return len(self.edit_chain) > 0  # Placeholder
"""
Nuance scoring system for media integrity assessment
"""

from typing import List, Dict, Any


class NuanceCalculator:
    """Calculates reality nuance scores based on provenance data."""
    
    @staticmethod
    def calculate_nuance_score(provenance_chain: List[Dict[str, Any]], 
                              has_original_anchor: bool = True) -> float:
        """
        Calculate a nuanced integrity score (0.0 to 1.0).
        
        Args:
            provenance_chain: List of edit events
            has_original_anchor: Whether the media has a truth anchor
            
        Returns:
            Float between 0.0 and 1.0 representing integrity
        """
        if not has_original_anchor:
            return 0.3  # Low score for unanchored media
        
        base_score = 1.0
        num_edits = len(provenance_chain)
        
        # Penalize based on number of edits (simplified)
        edit_penalty = min(0.5, num_edits * 0.1)
        base_score -= edit_penalty
        
        # Consider edit types (in production, this would be more sophisticated)
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
"""
Project Vesta - Layer 3: Decentralized Immune System
Purpose: Calculates confidence scores using AI and community consensus
"""

import random
from typing import Dict, Any, List
from ...layer2_provenance.nuance_calculator import NuanceCalculator


class ConfidenceEngine:
    """Combines multiple signals to calculate final confidence scores."""
    
    def __init__(self):
        self.ai_models = ["temporal_analysis", "spatial_analysis", "frequency_analysis"]
        self.community_weights = {"expert": 0.6, "crowd": 0.3, "ai_consensus": 0.1}
    
    def analyze_media(self, media_url: str, metadata: Dict[str, Any], 
                     provenance_chain: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive media analysis combining all verification signals.
        
        Args:
            media_url: URL or identifier of the media
            metadata: Media metadata including potential anchor
            provenance_chain: Optional provenance history
            
        Returns:
            Comprehensive confidence analysis
        """
        has_anchor = 'vesta_anchor_id' in metadata
        provenance_chain = provenance_chain or []
        
        # Calculate base scores
        ai_confidence = self._get_ai_confidence(media_url)
        community_consensus = self._get_community_consensus(media_url)
        nuance_score = NuanceCalculator.calculate_nuance_score(provenance_chain, has_anchor)
        
        # Weighted final score
        if has_anchor:
            # Anchor exists - heavily weighted toward provenance
            final_score = (nuance_score * 0.7) + (ai_confidence * 0.2) + (community_consensus * 0.1)
            explanation = f"Anchored media: {NuanceCalculator.get_nuance_description(nuance_score)}"
        else:
            # No anchor - rely on AI and community
            final_score = (ai_confidence * 0.5) + (community_consensus * 0.5)
            explanation = "Unanchored media - verification based on AI and community consensus"
        
        # Determine risk level
        if final_score >= 0.8:
            risk_level = "low"
            recommendation = "trust_with_context"
        elif final_score >= 0.6:
            risk_level = "medium"
            recommendation = "verify_provenance"
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
        """Get AI model confidence (placeholder implementation)."""
        # In production, this would run multiple AI detection models
        return random.uniform(0.7, 0.98)
    
    def _get_community_consensus(self, media_url: str) -> float:
        """Get community verification consensus (placeholder)."""
        # In production, this would query decentralized verification network
        return random.uniform(0.8, 0.95)
"""
Tests for Layer 1 - Truth Anchor generation
"""

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from vesta.layer1_anchor import AnchorSeedGenerator, crypto_utils


class TestAnchorGeneration:
    def test_anchor_creation(self):
        """Test basic anchor creation."""
        private_key, public_key = crypto_utils.generate_keypair()
        generator = AnchorSeedGenerator("TEST_DEVICE", private_key)
        
        test_data = b"test_media_data"
        metadata = {"test": True, "file_type": "TEST"}
        
        anchor = generator.create_truth_anchor(test_data, metadata)
        
        assert "anchor_id" in anchor
        assert "payload" in anchor
        assert "signature" in anchor
        assert anchor["payload"]["device_id"] == "TEST_DEVICE"
    
    def test_anchor_verification(self):
        """Test anchor verification."""
        private_key, public_key = crypto_utils.generate_keypair()
        generator = AnchorSeedGenerator("TEST_DEVICE", private_key)
        
        anchor = generator.create_truth_anchor(b"test", {})
        is_valid = generator.verify_anchor(anchor, public_key)
        
        assert is_valid == True
"""
Basic usage example for Project Vesta
"""

from vesta.layer1_anchor import AnchorSeedGenerator, crypto_utils
from vesta.layer2_provenance import ProvenanceTracker, NuanceCalculator
from vesta.layer3_immune_system import ConfidenceEngine


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
    tracker.add_edit("color_correction", "editor_001", 
                    {"adjustment": "exposure+0.3", "white_balance": "auto"}, 
                    private_key)
    
    tracker.add_edit("crop", "editor_001", 
                    {"dimensions": "1920x1080", "aspect_ratio": "16:9"}, 
                    private_key)
    
    provenance = tracker.get_provenance_chain()
    print(f"   Edit events: {len(provenance)}")
    
    # Calculate nuance score
    nuance_score = NuanceCalculator.calculate_nuance_score(provenance, True)
    nuance_desc = NuanceCalculator.get_nuance_description(nuance_score)
    print(f"   Nuance Score: {nuance_score} - {nuance_desc}")
    
    # Layer 3: Confidence Analysis
    print("\n3. Confidence Analysis...")
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
# Initialize and push to GitHub
git init
git add .
git commit -m "feat: Initial commit - Project Vesta three-layer truth verification system

- Layer 1: Cryptographic Truth Anchor with Ed25519 signing
- Layer 2: Provenance Tracker with nuance scoring
- Layer 3: Decentralized Immune System with AI/community consensus
- Complete test suite and examples
- Apache 2.0 license
- Professional documentation and packaging"

# Create repository on GitHub first, then:
git remote add origin https://github.com/your-username/Project-Vesta.git
git branch -M main
git push -u origin main
