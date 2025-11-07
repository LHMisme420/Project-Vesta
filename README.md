# Project-Vesta
Generates the immutable digital birth certificate
# =======================================================
# Project Vesta - Layer 1: The Cryptographic Truth Anchor
# File: anchor_seed_generator.py
# Purpose: Generates the immutable digital birth certificate (Anchor ID).
# License: Apache 2.0 (Aligned with Vesta Ethical Charter)
# =======================================================

import hashlib
import time
# NOTE: In a real system, 'perceptual_hash_lib' would be a complex library
# for image/video hashing robust against minor compression.
# 'crypto_lib' would handle Zero-Knowledge-Proof-compatible key signing.

class AnchorSeedGenerator:
    """Creates a verifiable, signed Anchor for a piece of media."""
    
    def __init__(self, device_id: str, private_key: str):
        self.device_id = device_id
        self.private_key = private_key
        
    def generate_perceptual_hash(self, raw_data: bytes) -> str:
        """
        Placeholder for a robust, collision-resistant perceptual hash function.
        In I.P.P., this is AI-powered to resist minor manipulation.
        """
        # Simple placeholder hash for now (uses SHA-256 for integrity)
        return hashlib.sha256(raw_data).hexdigest()

    def create_truth_anchor(self, raw_data: bytes, metadata: dict) -> dict:
        """
        Main function to generate the signed Anchor object.
        """
        timestamp = int(time.time())
        perceptual_hash = self.generate_perceptual_hash(raw_data)
        
        # Data structure to be signed and logged
        signed_data_payload = {
            "p_hash": perceptual_hash,
            "device_id": self.device_id,
            "timestamp": timestamp,
            "metadata": metadata
        }
        
        # Placeholder for signature generation (must use ZKP-compatible crypto)
        signature = f"SIGNED_BY_{self.device_id}_AT_{timestamp}_HASH_{perceptual_hash[:8]}"
        
        anchor_id = hashlib.sha1(signature.encode('utf-8')).hexdigest()

        return {
            "anchor_id": anchor_id,
            "payload": signed_data_payload,
            "signature": signature
        }

# Example Usage:
# # This payload would be sent to the decentralized ledger for logging.
# generator = AnchorSeedGenerator("DEVICE_CAM_XYZ789", "MY_SUPER_SECRET_KEY")
# anchor = generator.create_truth_anchor(b"raw_image_data_bytes...", {"file_type": "JPEG"})
# print(anchor)// =======================================================
// Project Vesta - Layer 2: The Provenance Tracker
// File: provenance_viewer.js
// Purpose: Visualizes the verifiable edit history of a media file.
// =======================================================

/**
 * Renders the Provenance Timeline and the crucial Nuance Bar.
 * @param {HTMLElement} targetElement - The DOM element to render into.
 * @param {Array<Object>} provenanceChain - List of signed edit events.
 * @param {string} originalAnchorHash - The p_hash from the initial Anchor.
 * @param {string} currentMediaHash - The p_hash of the file being viewed.
 */
function renderProvenanceViewer(targetElement, provenanceChain, originalAnchorHash, currentMediaHash) {
    targetElement.innerHTML = `<h2>Provenance Timeline (${originalAnchorHash.substring(0, 8)}...)</h2>`;
    
    // --- Nuance Bar Logic (The Anti-Binary Tool) ---
    // A complex algorithm would calculate this, here we use a simple heuristic.
    const editScore = 1.0 - (provenanceChain.length * 0.1); // 10% reduction per major edit
    const hashIntegrity = calculateHashIntegrity(originalAnchorHash, currentMediaHash);
    const finalNuanceScore = Math.max(0.1, Math.min(1.0, (editScore + hashIntegrity) / 2.0));
    
    // Display the Nuance Bar
    targetElement.innerHTML += `
        <div style="margin-top: 15px;">
            <strong>Reality Nuance Score: ${Math.round(finalNuanceScore * 100)}%</strong>
            <div style="width: 100%; background: #ddd; height: 10px; border-radius: 5px;">
                <div style="width: ${finalNuanceScore * 100}%; background: #4CAF50; height: 10px; border-radius: 5px;"></div>
            </div>
            <small>(${finalNuanceScore > 0.8 ? 'High Integrity' : 'Requires Contextual Compassion'})</small>
        </div>
    `;

    // --- Timeline Rendering ---
    const timeline = document.createElement('ul');
    timeline.innerHTML = provenanceChain.map(event => `
        <li style="border-left: 3px solid #ccc; padding-left: 10px; margin-top: 5px;">
            <strong>${new Date(event.timestamp * 1000).toLocaleTimeString()}</strong>: 
            ${event.editType} by ${event.editorID.substring(0, 8)}...
            <span onclick="verifyEvent('${event.signedHash}')" style="cursor: pointer; color: blue;">[Verify]</span>
        </li>
    `).join('');

    targetElement.appendChild(timeline);
}

// Placeholder for real verification logic
function verifyEvent(signedHash) {
    // In I.P.P., this calls an API to check the hash against the immutable ledger.
    alert(`Checking ledger for hash: ${signedHash}. Status: VERIFIED.`);
}

function calculateHashIntegrity(originalHash, currentHash) {
    // Bounty #2: Developers must implement a sophisticated comparison function
    // that measures the *distance* between the perceptual hashes.
    return 0.8; // Placeholder: Assume 80% integrity for demo
}# =======================================================
# Project Vesta - Layer 3: The Decentralized Immune System
# File: immune_agent_logic.py
# Purpose: Calculates the final Confidence Score for any media viewed online.
# =======================================================

# NOTE: This module runs locally or on the decentralized verifier network.

def get_ai_model_confidence(media_url: str) -> float:
    """Placeholder for running local or remote AI Deepfake models."""
    # In I.P.P., multiple AI models (temporal, spatial, frequency-based) run in parallel.
    # Return a synthetic average score based on current market model performance.
    import random
    return random.uniform(0.75, 0.99) 

def get_community_consensus(media_hash: str) -> float:
    """Fetches consensus score from the crowdsourced verification network."""
    # This involves fetching votes from human and AI verifiers on the network.
    # Higher score = more consensus on the current state.
    return 0.90 # Placeholder consensus

def calculate_final_confidence(media_url: str, metadata: dict) -> dict:
    """
    Combines Anchor presence, AI analysis, and Community Consensus 
    to create the final, anti-binary Confidence Score.
    """
    has_anchor = 'vesta_anchor_id' in metadata
    
    if has_anchor:
        # If an Anchor exists, integrity is paramount (score based heavily on provenance)
        final_score = 0.95 
        explanation = f"Verified: Anchor found ({metadata['vesta_anchor_id'][:8]}...). Check Provenance Viewer for edits."
    else:
        # If no Anchor, rely on AI and Community for risk assessment.
        ai_score = get_ai_model_confidence(media_url)
        community_score = get_community_consensus(media_url)
        
        # Triangulation Point: Combine multiple non-ideological sources
        final_score = (ai_score + community_score) / 2.0
        
        if final_score < 0.80:
            explanation = "Caution: No Anchor. Community/AI confidence below threshold."
        else:
            explanation = "Unverified Source, but AI/Community Consensus is high."

    return {
        "confidence_score": final_score,
        "explanation": explanation,
        "has_anchor": has_anchor
    }

# Example Usage:
# # This is what the browser extension would call when loading a page.
# result = calculate_final_confidence("https://someimage.jpg", {})
# print(result)
import hashlib
import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519

class AnchorSeedGenerator:
    """Creates a verifiable, signed Anchor for a piece of media."""
    
    def __init__(self, device_id: str, private_key: ed25519.Ed25519PrivateKey):
        self.device_id = device_id
        self.private_key = private_key
        
    def generate_perceptual_hash(self, raw_data: bytes) -> str:
        """Enhanced perceptual hash with timestamp nonce"""
        timestamp_nonce = str(time.time()).encode()
        combined_data = raw_data + timestamp_nonce
        return hashlib.sha256(combined_data).hexdigest()
    
    def create_truth_anchor(self, raw_data: bytes, metadata: dict) -> dict:
        timestamp = int(time.time())
        perceptual_hash = self.generate_perceptual_hash(raw_data)
        
        payload = f"{perceptual_hash}|{self.device_id}|{timestamp}|{str(metadata)}"
        signature = self.private_key.sign(payload.encode())
        
        return {
            "anchor_id": hashlib.sha256(signature).hexdigest(),
            "payload": {
                "p_hash": perceptual_hash,
                "device_id": self.device_id,
                "timestamp": timestamp,
                "metadata": metadata
            },
            "signature": signature.hex()
        }
