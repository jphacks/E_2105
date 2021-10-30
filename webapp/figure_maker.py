import pathlib
import pandas as pd
from fetch_arxiv import fetch_search_result
from preprocessing_of_words import make_bow
import numpy as np
import pickle
import plotly.graph_objects as go
from jax_tsom import ManifoldModeling as MM
from sklearn.decomposition import NMF
from scipy.spatial import distance as dist
from Grad_norm import Grad_Norm
from webapp import logger


resolution = 10
u_resolution = 10
word_num = 200
CCP_VIEWER = 'CCP'
UMATRIX_VIEWER = 'U-matrix'
TOPIC_VIEWER = 'topic'
ycolors =  [[255, 255, 230], [255, 255, 180], [255, 253, 140], [255, 250, 115], [255, 235, 80], [231, 223, 37], [210, 200, 5], [155, 148, 15]]
gcolors = [[255, 255, 230], [255, 255, 180], [230, 247, 155], [211, 242, 132], [180, 220, 110], [144, 208, 80], [120, 180, 75]]
PAPER_COLORS = list(map(lambda i: 'rgb({},{},{})'.format(*i), gcolors))
WORD_COLORS = list(map(lambda i: 'rgb({},{},{})'.format(*i), ycolors))
PAPER_COLOR = PAPER_COLORS[3]
WORD_COLOR = WORD_COLORS[3]


def prepare_umatrix(keyword, X, Z1, Z2, sigma, labels, u_resolution, within_5years):
    within_5years_sign = '_within5y' if within_5years else ''
    umatrix_save_path = 'data/tmp/'+ keyword + within_5years_sign + '_umatrix_history.pickle'
    if pathlib.Path(umatrix_save_path).exists():
        logger.debug("U-matix already calculated")
        with open(umatrix_save_path, 'rb') as f:
            umatrix_history = pickle.load(f)
    else:
        logger.debug("Umatrix calculating")
        umatrix = Grad_Norm(
            X=X,
            Z=Z1,
            sigma=sigma,
            labels=labels,
            resolution=u_resolution,
            title_text="dammy"
        )
        U_matrix1, _, _ = umatrix.calc_umatrix()
        umatrix2 = Grad_Norm(
            X=X.T,
            Z=Z2,
            sigma=sigma,
            labels=labels,
            resolution=u_resolution,
            title_text="dammy"
        )
        U_matrix2, _, _ = umatrix2.calc_umatrix()
        umatrix_history = dict(
            umatrix1=U_matrix1.reshape(u_resolution, u_resolution),
            umatrix2=U_matrix2.reshape(u_resolution, u_resolution),
            zeta=np.linspace(-1, 1, u_resolution),
        )
        logger.debug("Calculating finished.")
        with open(umatrix_save_path, 'wb') as f:
            pickle.dump(umatrix_history, f)

    return umatrix_history


def prepare_materials(keyword, model_name, within_5years):
    logger.info(f"Preparing {keyword} map with {model_name}")
    base_filename = f"{keyword}{'_within5y' if within_5years else ''}"

    # Learn model
    nb_epoch = 50
    sigma_max = 2.2
    sigma_min = 0.2
    tau = 50
    latent_dim = 2
    seed = 1

    # Load data
    if pathlib.Path(f"{base_filename}.csv").exists():
        logger.debug("Data exists")
        csv_df = pd.read_csv(f"{base_filename}.csv")
        paper_labels = csv_df['site_name']
        rank = csv_df['ranking']
        X = np.load(f"data/tmp/{base_filename}.npy")
        word_labels = np.load(f"data/tmp/{base_filename}_label.npy")
    else:
        logger.debug("Fetch data to learn")
        csv_df = fetch_search_result(keyword, within_5years)
        paper_labels = csv_df['site_name']
        X , word_labels = make_bow(csv_df)
        rank = np.arange(1, X.shape[0]+1)  # FIXME
        csv_df.to_csv(f"{base_filename}.csv")
        feature_file = f'data/tmp/{base_filename}.npy'
        word_label_file = f'data/tmp/{base_filename}_label.npy'
        np.save(feature_file, X)
        np.save(word_label_file, word_labels)

    labels = (paper_labels, word_labels)
    model_save_path = f'data/tmp/{base_filename}_history.pickle'
    if pathlib.Path(model_save_path).exists():
        logger.debug("Model already learned")
        with open(model_save_path, 'rb') as f:
            history = pickle.load(f)
    else:
        logger.debug("Model learning")
        np.random.seed(seed)
        mm = MM(
            X,
            latent_dim=latent_dim,
            resolution=resolution,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            model_name=model_name,
            tau=tau,
            init='parafac'
        )
        mm.fit(nb_epoch=nb_epoch)
        history = dict(
            Z1=mm.history['z1'][-1],
            Z2=mm.history['z2'][-1],
            Y=mm.history['y'][-1],
            sigma=mm.history['sigma'][-1],
            Zeta=mm.Zeta1,
            resolution=mm.resoluton
        )
        logger.debug("Learning finished.")
        with open(model_save_path, 'wb') as f:
            pickle.dump(history, f)

    umatrix_history = prepare_umatrix(
        keyword,
        X,
        history['Z1'],
        history['Z2'],
        history['sigma'],
        None,
        u_resolution,
        within_5years,
    )
    return csv_df, labels, X, history, rank, umatrix_history


def draw_umatrix(fig, umatrix_history, viewer_id):
    if viewer_id == 'viewer_1':
        z = umatrix_history['umatrix1']
    elif viewer_id == 'viewer_2':
        z = umatrix_history['umatrix2']
    zeta = umatrix_history['zeta']
    fig.add_trace(
        go.Contour(
            x=zeta,
            y=zeta,
            z=z,
            name='contour',
            colorscale="gnbu",
            hoverinfo='skip',
            showscale=False,
        )
    )
    return fig


def draw_topics(fig, Y, n_components, viewer_id):
    # decomposed by Topic
    Y = Y.reshape(Y.shape[0], Y.shape[0])
    model_t3 = NMF(
        n_components=n_components,
        init='nndsvd',
        random_state=2,
        max_iter=300,
        solver='cd'
    )
    W = model_t3.fit_transform(Y)
    if viewer_id == 'viewer_2':
        W = model_t3.components_.T

    # For mask and normalization(min:0, max->1)
    mask_std = np.zeros(W.shape)
    mask = np.argmax(W, axis=1)
    for i, max_k in enumerate(mask):
        mask_std[i, max_k] = 1 / np.max(W)
    W_mask_std = W * mask_std
    DEFAULT_PLOTLY_COLORS = [
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)',
        'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
        'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
        'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
        'rgb(188, 189, 34)', 'rgb(23, 190, 207)'
    ]
    alpha = 0.1
    DPC_with_Alpha = [k[:-1] + ', ' + str(alpha) + k[-1:] for k in DEFAULT_PLOTLY_COLORS]
    for i in range(n_components):
        fig.add_trace(
            go.Contour(
                x=np.linspace(-1, 1, resolution),
                y=np.linspace(-1, 1, resolution),
                z=W_mask_std[:, i].reshape(resolution, resolution),
                name='contour',
                colorscale=[
                    [0, "rgba(0, 0, 0,0)"],
                    [1.0, DPC_with_Alpha[i]]],
                hoverinfo='skip',
                showscale=False,
            )
        )
    return fig


def draw_ccp(fig, Y, Zeta, resolution, clickedData, viewer_id):
    logger.debug('ccp')
    if viewer_id == 'viewer_1':
        y = Y[:, get_bmu(Zeta, clickedData)].reshape(resolution, resolution)
        colors = WORD_COLORS 
    elif viewer_id == 'viewer_2':
        y = Y[get_bmu(Zeta, clickedData), :].reshape(resolution, resolution)
        colors = PAPER_COLORS
    fig.add_trace(
        go.Contour(
            x=np.linspace(-1, 1, resolution),
            y=np.linspace(-1, 1, resolution),
            z=y,
            name='contour',
            colorscale=colors,
            hoverinfo='skip',
            showscale=False,
        )
    )
    return fig


def get_bmu(Zeta, clickData):
    clicked_point = [[clickData['points'][0]['x'], clickData['points'][0]['y']]] if clickData else [[0, 0]]
    clicked_point = np.array(clicked_point)
    dists = dist.cdist(Zeta, clicked_point)
    unit = np.argmin(dists, axis=0)
    return unit[0]


def draw_scatter(fig, Z, labels, rank, viewer_name):
    rank = np.linspace(1, len(labels), len(labels))
    logger.debug(f"viewer_name: {viewer_name}")
    logger.debug(f"Z: {Z.shape}, labels:{len(labels)}, rank:{len(rank)}")
    color = PAPER_COLORS[-1]
    if viewer_name == 'viewer_2':
        Z = Z[:word_num]
        labels = labels[:word_num]
        rank = rank[:word_num]
        color = WORD_COLORS[-1]

    fig.add_trace(
        go.Scatter(
            x=Z[:, 0],
            y=Z[:, 1],
            mode=f"markers+text",
            name="",
            marker=dict(
                size=(rank[::-1])*(1 if viewer_name == 'viewer_1' else 0.5),
                sizemode='area',
                sizeref=2. * max(rank) / (40. ** 2),
                sizemin=10,
            ),
            text=(labels if viewer_name == 'viewer_2' else rank),
            textfont=dict(
                family="sans serif",
                size=10,
                color='black'
            ),
            hovertext=labels,
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.75)",
            ),
            textposition='top center',
            hovertemplate="<b>%{hovertext}</b>",
        )
    )
    # fig.add_annotation(
    #     x=Z[:, 0],
    #     y=Z[:, 1],
    #     text=(labels if viewer_name == 'viewer_2' else list(map(lambda i: str(i), rank))),
    #     showarrow=False,
    #     yshift=10)
    return fig


def make_figure(history, umatrix_hisotry, X, rank, labels, viewer_name='U_matrix', viewer_id=None, clicked_z=None):
    logger.debug(viewer_id)
    if viewer_id == 'viewer_1':
        Z, Y = history['Z1'], history['Y']
        labels = labels[0] if isinstance(labels[0], list) else labels[0].tolist()
    elif viewer_id == 'viewer_2':
        Z, Y = history['Z2'], history['Y']
        X = X.T
        labels = labels[1] if isinstance(labels[1], list) else labels[1].tolist()
        logger.debug(f"LABELS: {labels[:5]}")
    else:
        logger.debug("Set viewer_id")

    # Build figure
    x1, x2 = Z[:, 0].min(), Z[:, 0].max()
    y1, y2 = Z[:, 1].min(), Z[:, 1].max()
    fig = go.Figure(
        layout=go.Layout(
            xaxis=dict(
                range=[Z[:, 0].min() + 0.05, Z[:, 0].max() + 0.05],
                visible=False,
                autorange=True,
            ),
            yaxis=dict(
                range=[Z[:, 1].min() - 0.1, Z[:, 1].max() + 0.2],
                visible=False,
                scaleanchor='x',
                scaleratio=1.0,
            ),
            showlegend=False,
            autosize=True,
            plot_bgcolor="#FFFFFF",
            margin=dict(
                b=0,
                t=0,
                l=0,
                r=0,
            ),
        ),
    )

    if viewer_name == "topic":
        n_components = 5
        fig = draw_topics(fig, Y, n_components, viewer_id)
    elif viewer_name == "CCP":
        fig = draw_ccp(fig, Y, history['Zeta'], history['resolution'], clicked_z, viewer_id)
    else:
        fig = draw_umatrix(fig, umatrix_hisotry, viewer_id)
    if viewer_id == 'viewer_2':
        _, unique_Z_idx = np.unique(Z, axis=0, return_index=True)
        logger.debug(unique_Z_idx)
        duplicated_Z_idx = np.setdiff1d(np.arange(Z.shape[0]), unique_Z_idx)
        labels = np.array(labels)
        labels[duplicated_Z_idx] = ''

    fig = draw_scatter(fig, Z, labels, rank, viewer_id)

    fig.update_coloraxes(
        showscale=False
    )
    fig.update_layout(
        plot_bgcolor=(PAPER_COLOR if viewer_id == 'viewer_1' else WORD_COLOR),
    )


    fig.update(
        layout_coloraxis_showscale=False,
        layout_showlegend=False,
    )
    fig.update_yaxes(
        fixedrange=True,
    )
    fig.update_xaxes(
        fixedrange=True,
    )

    return fig


def make_first_figure(viewer_id):
    _, labels, X, history, rank, umatrix_hisotry = prepare_materials('Machine Learning', 'TSOM', False)
    return make_figure(history, umatrix_hisotry, X, rank, labels, 'U-matrix', viewer_id, None)


def draw_toi(fig, clickData, view_method, viewer_id):
    if not clickData:
        return fig

    color = {
        CCP_VIEWER: 'green',
        UMATRIX_VIEWER: '#ffd700',
        TOPIC_VIEWER: 'yellow',
    }[view_method]
    color = PAPER_COLORS if viewer_id == 'viewer_1' else WORD_COLORS
    x, y = clickData['points'][0]['x'], clickData['points'][0]['y']
    radius = 0.15
    fig.add_shape(
        type='circle',
        line=dict(
            color=color[0],
            width=9.0,
        ),
        x0=(x - radius),
        y0=(y - radius),
        x1=(x + radius),
        y1=(y + radius),
    )
    fig.add_shape(
        type='circle',
        line=dict(
            color=color[-1],
            width=5,
        ),
        x0=(x - radius),
        y0=(y - radius),
        x1=(x + radius),
        y1=(y + radius),
    )
    return fig
