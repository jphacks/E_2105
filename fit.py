# import sys
# sys.path.append('../')
from som import ManifoldModeling as MM
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import pickle
from sklearn.decomposition import NMF

def update(i, fig_title, Z):
    if i != 0:
        plt.cla()                      # 現在描写されているグラフを消去
    plt.scatter(Z[i, :, 0], Z[i, :, 1])
    plt.title(fig_title + 'i=' + str(i))


if __name__ == '__main__':
    from dev.Grad_norm_dev import Grad_Norm

    keyword = "ファッション"
    model = "SOM"
    feature_file = 'data/tmp/'+keyword+'.npy'
    label_file = 'data/tmp/'+keyword+'_label.npy'
    X = np.load(feature_file)
    labels_long = np.load(label_file)
    labels = ["{:.8}".format(label.replace(keyword,'')) for label in labels_long]
    print(labels)

    nb_epoch = 500
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.2
    tau = 50
    latent_dim = 2
    seed = 1

    title_text= "animal map"
    umat_resolution = 100 # U-matrix表示の解像度

    np.random.seed(seed)

    mm = MM(X, model_name='SOM', latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau, init='PCA')
    mm.fit(nb_epoch=nb_epoch)

    mm_umatrix = Grad_Norm(X=X,
                            Z=mm.history['z'],
                            sigma=mm.history['sigma'],
                            labels=labels,
                            resolution=umat_resolution,
                            title_text=title_text)
    # mm_umatrix.draw_umatrix()
    Z = mm.history['z'][-1]
    Y = mm.history['y'][-1]
    model_t3 = NMF(n_components=5, init='random', random_state=2, max_iter=300,
                           solver='cd')
    Wt3 = model_t3.fit_transform(Y)
    Ht3 = model_t3.components_

    fig = plt.figure(figsize=(16, 9))
    two = plt.imshow(Wt3[:, 1].reshape(resolution, resolution),
                                   extent=[Z[:, 0].min(), Z[:, 0].max(), Z[:, 1].min(),
                                           Z[:, 1].max()],
                                   interpolation=None, alpha=0.8)
    plt.show()

    
    # animation = ani.FuncAnimation(Fig, update, fargs=(np.zeros(10, 10, 2)))
    # ani = ani.FuncAnimation(fig, update, fargs = ('Initial Animation! ', mm.history['z']), \
    # interval = 1, frames = 500)
    # plt.show()


    # with open('data/tmp/'+keyword+'_'+model+'.pickle', 'wb') as f:
    #     pickle.dump(som, f)
