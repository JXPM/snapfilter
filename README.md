# **Analyse et Explication Technique du Code**

## **1. Présentation du Programme**
Bon, après avoir passé des heures à essayer de me coller une barbe virtuelle (spoile alerte : c’était un échec cuisant), je me suis donc reconcentré sur la 1ère idée : du rouge à lèvres. Ce programme applique donc un filtre réaliste de rouge à lèvres sur une personne détectée par la webcam. 

- On détecte le visage grâce à MediaPipe (et croyez-moi, à force, je connais tout les points du visage par cœur).
- On isole les lèvres.
- On applique une teinte personnalisée.
- On affiche le tout, histoire d’admirer le travail (ou de rigoler si l’algorithme déconne).

## **2. Bibliothèques Utilisées**
### **OpenCV (cv2)**
- Capture la vidéo, traite les images et affiche le résultat. Sans lui, on coderait à l’aveugle.

### **MediaPipe (mp.solutions.face_mesh)**
- Détecte les points clés du visage. Utile quand on veut éviter de colorier le nez en rouge par erreur.

### **NumPy (np)**
- Manipule les coordonnées des points des lèvres, parce qu’on n’a pas que ça à faire à calculer tout à la main.

## **3. Explication du Code**

### **3.1 Initialisation**
```python
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)
```
- On initialise le modèle de détection de visage. Pas de panique, il ne va pas confondre les yeux avec la bouche (enfin, en théorie).

```python
cap = cv2.VideoCapture(0)
cv2.namedWindow("Webcam - Rouge à Lèvres Réaliste", cv2.WINDOW_NORMAL)
```
- On active la webcam et on crée une fenêtre pour admirer le résultat en direct.

### **3.2 Définition des Points Clés des Lèvres**
```python
outer_lips_landmarks = [61, 146, 91, 181, ...]
inner_lips_landmarks = [78, 95, 88, 191, ...]
```
- Si vous vous demandez comment je les connais, dites-vous juste que j’ai passé BEAUCOUP trop de temps à analyser des visages.

### **3.3 Paramètres de Filtrage**
```python
lip_hue = 170  # Teinte rouge
lip_saturation = 200  # Intensité
alpha = 0.75  # Transparence du filtre
```
- Ajustables à souhait. Libre à vous de vous peindre les lèvres en bleu si ça vous chante.

### **3.4 Capture et Traitement Vidéo**
```python
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
```
- Boucle principale : on récupère l’image en direct, sinon on pleure et on quitte.

### **3.5 Détection et Transformation des Points Clés**
```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(frame_rgb)
```
- Conversion en RGB, parce que MediaPipe a des goûts bizarres en matière de couleurs.
- On envoie l’image au détecteur de visages.

### **3.6 Extraction des Coordonnées des Lèvres**
```python
outer_lips_coords = np.array([...], np.int32)
inner_lips_coords = np.array([...], np.int32)
```
- On convertit les points en coordonnées d’image. Plus rapide que de tout tracer à la main.

### **3.7 Création du Masque des Lèvres**
```python
lips_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(lips_mask, [cv2.convexHull(outer_lips_coords)], 255)
cv2.fillPoly(lips_mask, [cv2.convexHull(inner_lips_coords)], 0)
lips_mask = cv2.GaussianBlur(lips_mask, (5, 5), 3)
```
- On fabrique un masque qui ne garde que les lèvres.
- Un petit flou gaussien pour éviter d’avoir un effet peinture à la truelle.

### **3.8 Application de la Teinte Rouge**
```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
hsv[..., 0] = np.where(lips_mask == 255, lip_hue, hsv[..., 0])
hsv[..., 1] = np.where(lips_mask == 255, lip_saturation, hsv[..., 1])
```
- Magie du HSV : on change uniquement la teinte et la saturation des pixels concernés.

### **3.9 Fusion avec l'Image Originale**
```python
colored_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
frame = cv2.addWeighted(frame, 1 - alpha, colored_frame, alpha, 0)
```
- On fusionne l’effet avec l’image originale. Résultat : un rouge à lèvres plus ou moins naturel (ou pas).

### **3.10 Affichage et Quitter**
```python
cv2.imshow("Webcam - Rouge à Lèvres Réaliste", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```
- Appuyez sur 'q' pour quitter. Ou restez hypnotisé par votre reflet, c’est vous qui voyez.

### **3.11 Libération des Ressources**
```python
cap.release()
cv2.destroyAllWindows()
```
- Histoire de ne pas laisser la webcam tourner indéfiniment (on évite les films d’horreur IRL).

## **4. Améliorations Possibles**
### **4.1 Ajustement Dynamique de la Teinte**
- Parce que tout le monde n’a pas envie de se maquiller en vampire.

### **4.2 Détection Multi-Visages**
- Pour que tout le monde puisse en profiter, pas seulement la personne devant la caméra.

### **4.3 Amélioration des Contours des Lèvres**
- Histoire que l’algorithme ne confonde pas les lèvres avec la peau du menton (oui, c’est du vécu).

## **5. Conclusion**
Ce programme est un bon point de départ pour la réalité augmentée en cosmétique. Je voulais à la base faire une barbe virtuelle, mais je me suis vite rendu compte que c’était une galère sans nom. Les lèvres, au moins, sont plus faciles à isoler. Bref, amusez-vous bien à tester des rouges improbables !

