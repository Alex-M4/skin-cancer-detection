# Prédiction de cancer de la peau – HAM10000 🩺  

## 📌 Description du projet  

Ce projet vise à classifier des lésions cutanées en **deux classes** :  
- **Bénign**  
- **Cancéreux**  

à partir d’images dermatoscopiques issues du dataset **HAM10000**.
L’objectif est de construire un premier modèle de deep learning, d’équilibrer les données (≈ 60 % bénin / 40 % cancéreux) et d’évaluer ses performances sur cette tâche de classification binaire.

---

## 📂 Jeu de données – HAM10000  

Le dataset **HAM10000** (*Human Against Machine with 10000 training images*) contient **10 015 images dermatoscopiques** de lésions pigmentées de la peau, réparties initialement en **7 classes** (akiec, bcc, bkl, df, mel, nv, vasc).

Dans ce projet :  

- Les 7 classes ont été **regroupées en 2 classes** :  
  - **Bénin** : regroupement des lésions non cancéreuses (par ex. nevi, kératoses bénignes, etc.).  
  - **Cancéreux** : regroupement des lésions malignes (par ex. mélanome, carcinome basocellulaire…).  
- Un **rééquilibrage** des données a été réalisé pour obtenir une proportion d’environ **60 % bénin / 40 % cancéreux**, afin de limiter le déséquilibre et obtenir des premiers résultats plus stables.

Les images sont redimensionnées (par ex. 224×224) et normalisées avant d’être passées au modèle de deep learning.

---

## 🧠 Modèle et approche  

Le projet s’appuie sur un modèle de deep learning via le transfert learnig à partir du modèle pré-entrainer le RESNET 50 pour effectuer la classification binaire **bénin vs cancéreux**.  

Pipeline global :  
1. Chargement des images et des labels HAM10000.  
2. Regroupement des 7 classes en 2 labels (bénin / cancéreux).  
3. Séparation **train / validation / test**.  
4. **Rééquilibrage** des classes (≈ 60/40) par sous‑échantillonnage / sur‑échantillonnage ou augmentation de données.  
5. Entraînement des modèles de deep learning **modèle frozen** et **fine-tuning**.  
6. Évaluation sur le jeu de test **Recall**. Compte tenu de l'enjeu vital de ne pas manquer de cancers cutanés, le recall a été choisi pour évaluer la performance des modèles.
