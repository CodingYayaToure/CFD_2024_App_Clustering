import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Configurer la mise en page de la page pour être plus large
st.set_page_config(layout="wide")

# Informations personnelles
photo_url = "CFD_2024.png"

# Afficher les scores de silhouette
st.sidebar.subheader("Scores de Silhouette")
if algo_clustering == "KMeans":
    st.sidebar.write(f"Score de Silhouette KMeans: {score_kmeans:.2f}")
elif algo_clustering == "DBSCAN":
    st.sidebar.write(f"Score de Silhouette DBSCAN: {score_dbscan:.2f}")
elif algo_clustering == "GMM":
    st.sidebar.write(f"Score de Silhouette GMM: {score_gmm:.2f}")
elif algo_clustering == "CAH (Clustering Hiérarchique)":
    st.sidebar.write(f"Score de Silhouette Hiérarchique: {score_cah:.2f}")

prenom = "Yaya"
nom = "Toure"
email = "yaya.toure@unchk.edu.sn"
whatsapps_url = "https://wa.me/message/GW7RWRW3GR4WN1"
linkedin_url = "https://www.linkedin.com/in/yaya-toure-8251a4280/"
github_url = "https://github.com/CodingYayaToure"
universite = "Université Numérique Cheikh Hamidou KANE (UN-CHK)"
formation = "Licence Analyse Numérique et Modélisation | Master Calcul Scientifique et Modélisation"
certificat = "Collecte et Fouille de Données (UADB-CNAM Paris) | 2024"

# Interface Streamlit
st.sidebar.title("Informations personnelles")
st.sidebar.image(photo_url, caption=f"{prenom} {nom}", width=390)
st.sidebar.write(f"Nom: {prenom} {nom}")
st.sidebar.write(f"Email: {email}")
st.sidebar.markdown(f"[WhatsApp]({whatsapps_url})")
st.sidebar.markdown(f"[LinkedIn]({linkedin_url})")
st.sidebar.markdown(f"[GitHub]({github_url})")
st.sidebar.write(f"{universite}")
st.sidebar.write(f"**Formations:** {formation}")
st.sidebar.write(f"**Certificat:** {certificat}")

# Titre de la page
st.title('Application Interactive d\'Analyse de Clustering')

# Section de téléchargement de fichier
uploaded_file = st.file_uploader("Téléchargez votre fichier de données (CSV)", type="csv")

if uploaded_file is not None:
    # Charger le dataset
    data = pd.read_csv(uploaded_file)
    
    # Créer une disposition en colonnes pour le dataset et d'autres composants
    col1, col2 = st.columns([2, 1])  # Ajuster les largeurs des colonnes (col1 est plus large)
    
    # Afficher le dataset dans une colonne plus large
    with col1:
        st.write("Aperçu du Dataset :")
        st.dataframe(data, height=600)  # Le tableau prend plus de place maintenant
        
        # Ajouter une description de l'utilisation de l'application
        st.markdown("""
        ### Comment utiliser l'application :
        
        1. **Télécharger un Dataset** : Commencez par télécharger votre dataset au format CSV. Le dataset doit contenir des valeurs numériques.
        2. **Choisir un Algorithme de Clustering** : Sélectionnez votre algorithme de clustering préféré dans les options de la barre latérale (KMeans, DBSCAN, GMM ou CAH).
        3. **Ajuster les Paramètres** : Selon l'algorithme choisi, vous pouvez ajuster des paramètres tels que le nombre de clusters, epsilon (pour DBSCAN), ou le nombre de composants (pour GMM).
        4. **Voir les Résultats** : Les résultats du clustering seront affichés dans des graphiques interactifs en 2D utilisant PCA pour réduire la dimensionnalité des données.
        5. **Scores de Silhouette** : Consultez les scores de silhouette pour évaluer la qualité des clusters. Un score de silhouette plus élevé indique des clusters mieux définis.
        6. **Nombre de Clusters** : La barre latérale affichera également le nombre de clusters et le nombre d'individus dans chaque cluster.
        
        ### Remarque :
        - Les graphiques de dispersion sont interactifs : vous pouvez zoomer/dézoomer et passer la souris sur les points pour voir les détails.
        - L'analyse de silhouette aide à déterminer le nombre optimal de clusters pour KMeans et d'autres algorithmes de clustering.
        """)

    # Normaliser et standardiser les données
    x = data.values
    x_normalisé = normalize(x)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(x_normalisé)

    # PCA pour la visualisation
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Options de la barre latérale
    st.sidebar.title("Options de Clustering")
    algo_clustering = st.sidebar.selectbox(
        "Sélectionnez l'Algorithme de Clustering",
        ("KMeans", "DBSCAN", "GMM", "CAH (Clustering Hiérarchique)")
    )

    # Fonction pour tracer des sous-graphiques de silhouette pour différentes valeurs de k
    def tracer_silhouette_subplot(data, plage_n_clusters):
        fig, axes = plt.subplots(1, len(plage_n_clusters), figsize=(20, 8))  # Ajuster la taille du graphique
        for idx, n_clusters in enumerate(plage_n_clusters):
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            sample_silhouette_values = silhouette_samples(data, cluster_labels)

            ax = axes[idx]
            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                 0, ith_cluster_silhouette_values)
                y_lower = y_upper + 10

            ax.set_title(f"k={n_clusters}\nSilhouette Moyenne: {silhouette_avg:.2f}")
            ax.set_xlabel("Valeurs du coefficient de silhouette")
            ax.set_ylabel("Labels de Cluster")
        
        plt.tight_layout()
        st.pyplot(fig)

    # Fonction pour afficher les graphiques de dispersion et les résultats du clustering dans un DataFrame
    def afficher_resultats_clustering(labels, titre):
        # Graphique avec Plotly
        fig = px.scatter(
            x=data_pca[:, 0], y=data_pca[:, 1], color=labels,
            labels={'x': 'PCA1', 'y': 'PCA2'},
            title=titre,
            width=900, height=700
        )
        st.plotly_chart(fig)
        
        # Afficher le nombre d'individus dans chaque cluster dans un DataFrame
        classes_uniques, counts = np.unique(labels, return_counts=True)
        cluster_df = pd.DataFrame({
            'Cluster': classes_uniques,
            'Nombre d\'Individus': counts
        })
        st.write("Nombre d'individus par cluster :")
        st.dataframe(cluster_df)

    # Fonction pour afficher le dendrogramme pour CAH
    def afficher_dendrogramme(Z):
        fig, ax = plt.subplots(figsize=(10, 5))  # Taille du dendrogramme
        dendrogram(Z, ax=ax)
        plt.title("Dendrogramme pour le Clustering Hiérarchique")
        plt.xlabel("Points de données")
        plt.ylabel("Distance Euclidienne")
        st.pyplot(fig)

    # Afficher les résultats du clustering dans la deuxième colonne
    with col2:
        if algo_clustering == "KMeans":
            # Clustering KMeans
            k = st.sidebar.slider("Choisissez le nombre de clusters (k)", 2, 10, 4)
            kmeans = KMeans(n_clusters=k, random_state=42)
            
            labels_kmeans = kmeans.fit_predict(data_scaled)
            score_kmeans = silhouette_score(data_scaled, labels_kmeans)
            
            # Tracer les clusters KMeans et afficher les résultats dans un DataFrame
            afficher_resultats_clustering(labels_kmeans, f"KMeans Clustering (k={k}), Score de Silhouette: {score_kmeans:.2f}")

            # Afficher les sous-graphiques de silhouette pour une plage de k
            plage_n_clusters = [2, 3, 4, 5, 6]  # Ajuster si nécessaire
            tracer_silhouette_subplot(data_scaled, plage_n_clusters)

        elif algo_clustering == "DBSCAN":
            # Clustering DBSCAN
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
            min_samples = st.sidebar.slider("Nombre minimum d'échantillons", 1, 20, 5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels_dbscan = dbscan.fit_predict(data_scaled)
            
            if len(set(labels_dbscan)) > 1:
                score_dbscan = silhouette_score(data_scaled, labels_dbscan)
            else:
                score_dbscan = -1
            
            # Tracer les clusters DBSCAN et afficher les résultats dans un DataFrame
            afficher_resultats_clustering(labels_dbscan, f"Clustering DBSCAN, Score de Silhouette: {score_dbscan:.2f}")

        elif algo_clustering == "GMM":
            # Clustering par Modèle de Mélange Gaussien (GMM)
            k = st.sidebar.slider("Choisissez le nombre de composants (k)", 2, 10, 4)
            gmm = GaussianMixture(n_components=k, random_state=42)
            labels_gmm = gmm.fit_predict(data_scaled)
            score_gmm = silhouette_score(data_scaled, labels_gmm)
            
            # Tracer les clusters GMM et afficher les résultats dans un DataFrame
            afficher_resultats_clustering(labels_gmm, f"Clustering GMM (k={k}), Score de Silhouette: {score_gmm:.2f}")

        elif algo_clustering == "CAH (Clustering Hiérarchique)":
            # Clustering Hiérarchique (CAH)
            k = st.sidebar.slider("Choisissez le nombre de clusters", 2, 10, 4)
            Z = linkage(data_scaled, method='ward')
            labels_cah = fcluster(Z, t=k, criterion='maxclust')
            score_cah = silhouette_score(data_scaled, labels_cah)
            
            # Tracer les clusters CAH et afficher les résultats dans un DataFrame
            afficher_resultats_clustering(labels_cah, f"Clustering Hiérarchique (k={k}), Score de Silhouette: {score_cah:.2f}")
            
            # Afficher le dendrogramme
            afficher_dendrogramme(Z)

    
