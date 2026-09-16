# Neuro-CX Prototype — Concepts and Architecture Mapping Guide

This document explains how this research prototype implements the core concepts of **Neuro-CX** for your MSc Data Science project. It maps the mathematical model, project structure, data representations, and interactive dashboard elements back to their respective real-world and biological analogies.

---

## 1. Core Paradigm: Static Recommenders vs. Adaptive Cognitive CX

Traditional personalization engines rely on **static batch-updated profiles** (e.g., segmenting a user as a "Tech Enthusiast" based on historical purchases). This prototype models a customer's **evolving cognitive state** in real-time. 

In this system, a customer's state is not a set of static tags, but a dynamic, multi-dimensional vector that shifts with every action (viewing, adding to cart, purchasing) and decays over time (forgetting stale interests).

---

## 2. The "Neuroplasticity" Analogy: Concept Mapping

The architecture uses terms inspired by neurological memory systems. Below is how the biological and psychological processes map to PyTorch deep learning implementations:

| Biological/Psychological Concept | Mathematical/DL Representation | Implementation Details in `src/model/neuro_cx_model.py` |
| :--- | :--- | :--- |
| **Cognitive State / Working Memory** | **GRU Hidden State Vector ($h_t$)** | A $128$-dimensional continuous vector maintaining the cumulative trace of past actions. |
| **Sensory Stimuli** | **Feature Embeddings** | Category/item ID, action type, dwell time, and signal weight projected into a dense vector space. |
| **Long-Term Potentiation (LTP) / Synaptic Strengthening** | **Reinforcement Gate ($\alpha$)** | A learned gating network that scales the hidden-state update by action weight: $w \in [0.3, 3.0]$ (e.g., Purchase updates the state $10\times$ more than a Bounce). |
| **Synaptic Decay / Passive Forgetting** | **Analytical Exponential Decay** | Exponential decay applied to the hidden state at each step: $e^{-\lambda / w}$, where $\lambda$ is governed by the `decay_half_life`. |
| **Active Forgetting / Memory Consolidation** | **Gated Decay Modulator** | A learned Sigmoid gate ($\text{decay\_gate}(h_t, w_t)$) that selectively determines which dimensions of the hidden state to preserve or discard. |

---

## 3. Data Representation Mapping (`src/data_pipeline`)

Since your raw dataset (`Ecommerce_Consumer_Behavior_Analysis_Data.csv`) is cross-sectional (one row per customer profile), the pipeline transforms it into a **sequential clickstream** to train the sequence-learning model.

*   **Customer Demographics & Profile (Age, Income, Intent)**:
    *   *System Representation*: Used as generation seeds to establish a customer's **Preferred Category** and **Funnel Dynamics**.
*   **Need-Based vs. Impulsive Purchase Intent**:
    *   *System Representation*: Controls the sequence length and event mixture. *Impulsive* intent generates short view-to-purchase funnels; *Need-Based* intent generates long research funnels with multiple `view` and `add_to_cart` steps.
*   **Action Types**:
    *   `view` (Base stimulus, weight = 1.0)
    *   `add_to_cart` (Strong intent stimulus, weight = 1.5)
    *   `purchase` (Peak positive reinforcement stimulus, weight = 3.0)
    *   `review` (Post-purchase feedback reinforcement, weight = 1.5)
    *   `bounce` (Weak or negative stimulus, weight = 0.3)
*   **Dwell Time**:
    *   *System Representation*: Represents cognitive processing load. Normalised relative to 99th-percentile sessions to scale input density.

---

## 4. Model Architecture: Baseline vs. Neuro-CX

To prove the utility of the Neuro-CX mechanisms, the system contains two models for an **ablation study**:

```
1. Vanilla GRU Baseline (gru_baseline.py)
   Input Features ──> GRU Cell ──> Hidden State Update (Constant Weight) ──> Output Logits

2. Neuro-CX Model (neuro_cx_model.py)
   Input Features ──> GRU Cell ──> [Reinforcement Gate] ──> [Decay modulators] ──> Output Logits
                                           ▲                      ▲
                                     Action Weight           Half-life Decay
```

*   **Baseline Model**: Updates the hidden state uniformly. A view, a bounce, and a purchase all trigger equivalent transitions of the hidden state vector.
*   **Neuro-CX Model**: A purchase forces a massive state update, shifting recommendations immediately. A bounce triggers minimal state update and subjects the existing state vector to heavy exponential decay (forgetting).

---

## 5. Dashboard Interface: What Each Element Represents

When running the dashboard (`py run_app.py`), the interface visually demonstrates the cognitive adaptation theory:

```
┌────────────────────────────────────────────────────────────────────────┐
│  🧠 Neuro-CX Dashboard                                                 │
├───────────────────────────────┬────────────────────────────────────────┤
│  👤 Sidebar & Selector        │  🎯 Current Recommendations            │
│  - Select Customer            │  Ranked list of categories & items     │
│  - Select Model (Neuro-CX/Base)│  showing predicted probabilities       │
├───────────────────────────────┼────────────────────────────────────────┤
│  🎬 Sequence Stepper          │  🧬 Hidden State Evolution             │
│  [ Prev ]  Step 4/12  [ Next ]│  - Heatmap showing 64 activations      │
│  Action: ADD_TO_CART          │  - Mean activation line over time      │
└───────────────────────────────┴────────────────────────────────────────┘
```

### A. Customer Selector (Sidebar / Top Panel)
*   *What it represents*: Different consumer personas. Selecting a customer loads their synthetic clickstream history generated from their real survey response attributes.

### B. Sequence Stepper (Step 1 → Step N)
*   *What it represents*: A live, real-time browsing session. Clicking **Next** simulates the user performing a new action in your store.

### C. Current Recommendations (Left Column)
*   *What it represents*: The output of the cognitive state. After each step, the hidden state vector is fed through a fully connected output layer to predict the likelihood of the next item. The ranked list updates immediately.
*   *Demo behavior*: Notice how under **Neuro-CX**, a single `purchase` instantly shifts the recommendations to align with that category. Under **Baseline**, the shift is sluggish and takes multiple steps.

### E. Hidden State Heatmap (Right Column)
*   *What it represents*: The internal memory of the "customer's brain". 
    *   *The X-axis* represents time steps.
    *   *The Y-axis* represents dimensions of the $128$-dimensional hidden state.
    *   *Color intensity (Blue to Red)* represents neuron activation levels.
    *   *Demo behavior*: Watch how a `purchase` (high weight) lights up the columns in bright red/blue. When consecutive `view` or `bounce` steps happen, the color washes out toward white (neutral), showing the **exponential forgetting mechanism** actively dampening the memory of older categories.

### F. Mean Hidden Activation Line (Right Column)
*   *What it represents*: Cumulative interest intensity.
    *   Spikes correspond to high-weight events (e.g., Purchases marked with a ⭐).
    *   Gradual downward slopes correspond to decay (forgetting) during periods of passive browsing or bouncing.

### G. Performance Metrics Panel (Bottom Panel)
*   *What it represents*: The academic validation of the model.
    *   **Accuracy@1**: Next-item prediction precision. Neuro-CX achieves higher Accuracy@1 because reinforcement forces high confidence on immediate interests.
    *   **NDCG@k & HitRate@k**: Measures how well the models rank relevant items within the top $k$. Baseline often wins on wider lists (better recall) because it doesn't decay old context, whereas Neuro-CX trades breadth for highly focused top-1 precision.
