import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image, ImageDraw, ImageFilter
import random

class SimpleFruitClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.class_names = ['apple', 'banana', 'orange']
    
    def create_sample_data(self, n=100):
        colors = {
            'apple': {'r': (150, 255), 'g': (0, 100), 'b': (0, 100)},
            'banana': {'r': (200, 255), 'g': (200, 255), 'b': (0, 100)},
            'orange': {'r': (200, 255), 'g': (100, 200), 'b': (0, 50)}
        }
        X, y = [], []
        for i, fruit in enumerate(self.class_names):
            for _ in range(n):
                r = np.clip(random.randint(*colors[fruit]['r']) + random.randint(-20, 20), 0, 255)
                g = np.clip(random.randint(*colors[fruit]['g']) + random.randint(-20, 20), 0, 255)
                b = np.clip(random.randint(*colors[fruit]['b']) + random.randint(-20, 20), 0, 255)
                X.append([r, g, b, r+g+b, max(r,g,b)-min(r,g,b), r/(g+1), (r+b)/(g+1), abs(r-g), abs(g-b), abs(r-b)])
                y.append(i)
        return np.array(X), np.array(y)
    
    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(f"Train Accuracy: {self.model.score(X_train, y_train):.2f}")
        print(f"Test Accuracy: {self.model.score(X_test, y_test):.2f}")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.show()
    
    def feature_importance(self):
        names = ['R', 'G', 'B', 'Brightness', 'Contrast', 'R/G', '(R+B)/G', 'R-G', 'G-B', 'R-B']
        imp = self.model.feature_importances_
        idx = np.argsort(imp)[::-1]
        plt.bar([names[i] for i in idx], imp[idx])
        plt.title("Feature Importance")
        plt.xticks(rotation=45)
        plt.show()
    
    def create_synthetic_image(self, fruit):
        base = {'apple': (220, 50, 50), 'banana': (255, 255, 80), 'orange': (255, 165, 0)}[fruit]
        img = Image.new('RGB', (100, 100), base)
        draw = ImageDraw.Draw(img)
        c = tuple(max(0, x - 30) for x in base)
        draw.ellipse([25, 25, 75, 75], fill=c)
        return img.filter(ImageFilter.GaussianBlur(0.5))

    def extract_features_from_image(self, img):
        arr = np.array(img)
        r, g, b = np.mean(arr[:, :, 0]), np.mean(arr[:, :, 1]), np.mean(arr[:, :, 2])
        return np.array([[r, g, b, r+g+b, max(r,g,b)-min(r,g,b), r/(g+1), (r+b)/(g+1), abs(r-g), abs(g-b), abs(r-b)]])
    
    def predict_image(self, img):
        features = self.extract_features_from_image(img)
        pred = self.model.predict(features)[0]
        conf = self.model.predict_proba(features)[0][pred]
        return self.class_names[pred], conf

    def test_predictions(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, fruit in enumerate(self.class_names):
            img = self.create_synthetic_image(fruit)
            pred, conf = self.predict_image(img)
            axes[i].imshow(img)
            axes[i].set_title(f'{fruit}\nPredicted: {pred}\nConf: {conf:.2f}')
            axes[i].axis('off')
        plt.show()

def main():
    clf = SimpleFruitClassifier()
    X, y = clf.create_sample_data(200)
    clf.train_model(X, y)
    clf.feature_importance()
    clf.test_predictions()

if __name__ == "__main__":
    main()
