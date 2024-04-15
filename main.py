import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn import datasets
from sklearn.decomposition import PCA
def plot2D(matrix):
    plt.scatter(matrix[:, 0], matrix[:, 1], color='blue')
    plt.title('Wykresik 2D')
    plt.xlabel('Oś X')
    plt.ylabel('Oś Y')
    plt.show()

def plot1D(matrix):
    plt.scatter(matrix[:, 0], matrix[:, 0], color='blue')
    plt.title('Wykresik 1D')
    plt.xlabel('Oś X')
    plt.ylabel('Oś Y')
    plt.show()


def wiPCA(matrix, dest):
    #standaryzacja
    matrix = (matrix - np.mean(matrix,axis=0))        #wersja druga ktora nie generuje błędu
    cov_matrix = np.cov(matrix,rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    #posortuj eigenvalues, a nastepnie posortuj do nowych tablic egigvals i eigvecs posortowane
    sort_indexes = np.argsort(eigvals)[::-1]
    sort_eigvals = eigvals[sort_indexes]
    sort_eigvecs = eigvecs[:,sort_indexes]

    #do jakich wymiarow
    n_components = dest
    #wybierz takie eigenvals które zredukuja to do n-tego wymiaru
    #i.e. eigvals i eigvecs to jest cos co przetrzymuje w jaki sposob ztransformować dane w matrixie
    #żeby uzyskać nasz matrix oryginalny
    #więc wybierając od poczatku redukujemy je do potrzebnych nam wymiarow
    #nie umiem tego inaczej wytlumaczyc sobie xD
    eigenvecs_subset = sort_eigvecs[:,0:n_components]

    #transformacja danychhhhhhh
    data_transformed = np.dot(eigenvecs_subset.transpose(),matrix.transpose()).transpose()
    return data_transformed,sort_eigvecs,sort_eigvals,matrix,eigenvecs_subset

def inverse_PCA(data, pca_components):
    #data - dane po redukcji PCA

    #pca_components - macierz principal components
    data_reconstructed = np.dot(data, pca_components.T)
    return data_reconstructed


if __name__ == '__main__':
    #wygeneruj 200 2 wymiarowych zmiennych
    losowe = np.random.rand(200,2)

    losowe_reduced,losowe_reduced_eigvecs,losowe_reduced_eigvals,stand_losowe,losowe_eigvec_subset = wiPCA(losowe, 1)
    plot2D(losowe)
    plot1D(losowe_reduced)


    #porownanie sklearn pca do wipca
    pca = PCA(n_components=1)
    Xr = pca.fit(losowe).transform(losowe)
    #plot1D(Xr)

    #iris do 2d
    iris = datasets.load_iris()
    irisdf = pd.DataFrame(data=np.c_[iris['data'], iris['target']],columns=iris['feature_names'] + ['target'])
    x = irisdf.iloc[:,0:4]
    target = irisdf.iloc[:,4]
    #print(irisdf)
    pca=PCA(n_components=2)
    mat_iris2d = pca.fit(x).transform(x)
    mat_iris2dWi,mat_iris2d2_eigvecs,mat_iris2d2_eigvals,mat_iris2d_stdmat,mat_iris2d_eigvecs_subset = wiPCA(x,2)

    reduced_iris = pd.DataFrame(mat_iris2d,columns = ['PC1','PC2'])
    reduced_iris = pd.concat([reduced_iris,pd.DataFrame(irisdf.iloc[:,0:4])], axis=1)

    reduced_irisWi = pd.DataFrame(mat_iris2dWi, columns=['PC1', 'PC2'])
    reduced_irisWi = pd.concat([reduced_irisWi, pd.DataFrame(irisdf.iloc[:, 0:4])], axis=1)
    #plot irisa
    plt.figure(figsize=(6,6))
    sb.scatterplot(reduced_iris,x='PC1',y='PC2',hue=target,s=60)
    #plt.show()

    #plot irisa Wi
    plt.figure(figsize=(6,6))
    sb.scatterplot(reduced_irisWi,x='PC1',y='PC2',hue=target,s=60)
    #plt.show()
    #jest poprawnie, ale jakby przemnożone przez -1 czy coś, jest odbite po osi y :(

    #digits
    digits=datasets.load_digits()
    digitsdf = pd.DataFrame(data=np.c_[digits['data'], digits['target']],columns=digits['feature_names'] + ['target'])

    xdigits = digitsdf.iloc[:,0:64]     #kolumna 65 zawiera jedynie etykiety
    target = digitsdf.iloc[:, 64]       #z tego co rozumiem to jest tablica która jest po prostu do odróżnienia zmiennych od siebie

    digits2d,digits2d_eigvecs,digits2d_eigvals,digits2d_std_mat, digits2d_eigvecs_subset = wiPCA(xdigits,2)

    #print PC variation
    variations = np.sqrt((digits2d_eigvals/len(xdigits)**2))
    variations = np.cumsum(variations)
    exes = np.linspace(0,len(variations),len(variations))
    plt.figure(figsize=(6,6))
    plt.plot(exes,variations,color='blue')
    plt.xlabel('principal component index')
    plt.ylabel('accumulated variation') # Wydaje się że dobrze działa, jednak najwyższa wartość zamiast być = 1, jest równa 0.1
    plt.show()

    reduced_digitsWi = pd.DataFrame(digits2d, columns=['PC1', 'PC2'])
    reduced_digitsWi = pd.concat([reduced_digitsWi, pd.DataFrame(digitsdf.iloc[:, 0:64])], axis=1)
    plt.figure(figsize=(6, 6))
    sb.scatterplot(reduced_digitsWi, x='PC1', y='PC2', hue=target, s=60,palette='icefire')
    plt.show()

    #obliczanie błędu
    sums = []
    for i in range(1,65):
        digitsdf = pd.DataFrame(data=np.c_[digits['data'], digits['target']],
                                columns=digits['feature_names'] + ['target'])
        xdigits = digitsdf.iloc[:, 0:64]
        target = digitsdf.iloc[:,64]
        digits2d, digits2d_eigvecs, digits2d_eigvals, digits2d_std_mat, digits2d_eigvecs_subset = wiPCA(xdigits, i)
        digitsdf_data = digitsdf.drop(columns=['target'])   #bez ostatniej bo zakłamuje dane
        reconstr_data = np.array(inverse_PCA(digits2d,digits2d_eigvecs_subset))
        differences = np.sqrt(np.sum((digitsdf_data.to_numpy() - reconstr_data) ** 2))  #odleglość między punktami
        sums.append(np.mean(differences))


    #wykres błędu zależnie od zredukowanego rozmiaru
    sums=np.array(sums)
    exes = np.arange(1, len(sums) + 1)
    plt.figure(figsize=(6, 6))
    plt.plot(exes, sums, color='blue')
    plt.xlabel('Dimension')
    plt.ylabel('summed up data loss')
    plt.show()

    # obliczanie błędu dla funkcji normalnej
    sums = []
    for i in range(1, 65):
        digitsdf = pd.DataFrame(data=np.c_[digits['data'], digits['target']],
                                columns=digits['feature_names'] + ['target'])
        xdigits = digitsdf.iloc[:, 0:64]
        target = digitsdf.iloc[:, 64]
        pca = PCA(n_components=i)
        digits_pca= pca.fit(xdigits).transform(xdigits)
        digitsdf_data = digitsdf.drop(columns=['target'])  # bez ostatniej bo zakłamuje dane
        reconstr_data = pca.inverse_transform(digits_pca)
        differences = np.sqrt(np.sum((digitsdf_data.to_numpy() - reconstr_data) ** 2))  # odleglość między punktami
        sums.append(np.mean(differences))

    # wykres błędu zależnie od zredukowanego rozmiaru
    sums = np.array(sums)
    exes = np.arange(1, len(sums) + 1)
    plt.figure(figsize=(6, 6))
    plt.plot(exes, sums, color='blue')
    plt.xlabel('Dimension')
    plt.ylabel('summed up data loss')
    plt.show()