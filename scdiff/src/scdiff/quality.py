import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def rf_discriminability(X_real, X_sim, seed=42, 
                        test_size=0.2, n_estimators=100, max_depth=None,
                        pca_components=None):
    X = np.vstack([X_real, X_sim])
    y = np.concatenate([np.ones(X_real.shape[0]), np.zeros(X_sim.shape[0])])

    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=seed
    )
    
    clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)
    accuracy = accuracy_score(y_test, (p >= 0.5).astype(int))
    
    return auc, accuracy
    
def knn_discriminability(X_real, X_sim, seed=42, 
                         test_size=0.2, n_neighbors=10, 
                         pca_components=None):
    X = np.vstack([X_real, X_sim])
    y = np.concatenate([np.ones(X_real.shape[0]), np.zeros(X_sim.shape[0])])
    
    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
        print(f"PCA completed with {X.shape[1]} components")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p)
    accuracy = accuracy_score(y_test, (p >= 0.5).astype(int))
    
    return auc, accuracy