# ACC Migration Guide

## Purpose

This document explains how the banking fraud project can be translated into an **ACC-style fraud, waste, and abuse analytics capability**.

The goal is not to claim that banking fraud and ACC fraud are identical. They are not. The goal is to show that the **underlying fraud analytics operating model is highly transferable**.

---

## Shared analytics logic

Across banking and ACC-style fraud analytics, the common elements are:

- multi-source data matching
- entity and event-level risk scoring
- alert generation and prioritisation
- investigation or integrity review workflow
- outcome feedback into monitoring and retraining
- governance, privacy, and explainability requirements

---

## Entity mapping

| Banking project entity | ACC-style equivalent |
|---|---|
| customer | client |
| account | claim or entitlement record |
| transaction | payment, weekly compensation event, or service event |
| merchant / counterparty | provider |
| device / IP | identity or linkage signal |
| alert | integrity alert |
| investigation case | fraud review or integrity investigation |

---

## Use case mapping

### 1. Transaction fraud detection
**Banking:** suspicious payment events  
**ACC analogue:** suspicious payment or compensation events, including potential weekly compensation irregularity.

### 2. Account takeover detection
**Banking:** login/device behaviour suggests account compromise  
**ACC analogue:** suspicious claim access, identity misuse, or unusual claimant status changes.

### 3. Behaviour anomaly detection
**Banking:** customer/account behaviour changes sharply  
**ACC analogue:** claimant behaviour, claim progression, or service pathway deviates from expected patterns.

### 4. Merchant anomaly detection
**Banking:** suspicious merchant billing or counterparty risk  
**ACC analogue:** provider inappropriate billing, unusual service patterns, or overservicing.

### 5. Linked-entity / graph risk detection
**Banking:** fraud rings, shared devices, linked mule accounts  
**ACC analogue:** linked clients, providers, services, referrals, or repeated abnormal relationships across claims.

### 6. Investigation prioritisation
**Banking:** route high-value fraud alerts to investigators  
**ACC analogue:** rank integrity alerts for review teams based on likelihood, materiality, and actionability.

---

## Example translation of features

### Banking feature categories
- velocity and recency
- amount anomalies
- geo/device novelty
- peer-group deviation
- shared-entity counts
- linked-risk ratios

### ACC-style equivalents
- payment frequency deviation
- compensation amount deviation
- claimant status change velocity
- provider utilisation deviation
- service duration/frequency outliers
- linked client-provider-service counts
- prior review outcome ratios

---

## Example translation of labels

### Banking labels
- confirmed fraud
- account takeover confirmed
- funds recovered
- investigation substantiated
- high-priority alert accepted

### ACC-style labels
- integrity issue confirmed
- fraud / waste / abuse confirmed
- payment adjusted or recovered
- provider review upheld
- investigation or integrity review substantiated

---

## Transferability


> The exact business context changes, but the core analytics challenge is similar: identify suspicious entities and events, combine deterministic and statistical signals, prioritise limited review capacity, and operate within strong governance and explainability constraints.

---

## This matters for ACC

- weekly compensation vs earnings mismatch
- client entitlement irregularity
- provider billing anomaly
- overservicing or unusual service utilisation
- client/provider/service network risk
- integrity alert prioritisation

This repository provides a transferable blueprint for those problems because it already includes:

- event-level scoring
- entity-level scoring
- anomaly detection
- graph-based linked risk
- review prioritisation
- governance outputs

---

