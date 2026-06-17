import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import product

with open('blob_history.txt', 'r') as f: 
    lines = f.readlines() 
    lines = [line.split() for line in lines] 
    df = pd.DataFrame(lines, columns = ['iteration', 'neuron', 'signal']) 
    df['iteration'] = df['iteration'].astype(int) 
    df['signal'] = df['signal'].astype(float)
    df = df.drop_duplicates(subset=['iteration', 'neuron'])

# ----------------------------------------
# 1) Parse coordenadas neuron
# ----------------------------------------
def parse_coord(col):
    coords = col.str.split('-', expand=True).astype(int)
    coords.columns = ['x', 'y', 'z']
    return coords

coords = parse_coord(df['neuron'])
df = pd.concat([df.reset_index(drop=True), coords], axis=1)
print(df)

# ----------------------------------------
# 2) Definir limites e normalizar sinal
# ----------------------------------------
min_x, max_x = int(df['x'].min()), int(df['x'].max())
min_y, max_y = int(df['y'].min()), int(df['y'].max())
min_z, max_z = int(df['z'].min()), int(df['z'].max())

sig_min, sig_max = df['signal'].min(), df['signal'].max()

# ----------------------------------------
# 3) Grid de referência (neurônios inativos)
# ----------------------------------------
grid_points = np.array(list(product(range(min_x, max_x + 1),
                                    range(min_y, max_y + 1),
                                    range(min_z, max_z + 1))))
gx, gy, gz = grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]

bg_trace = go.Scatter3d(
    x=gx, y=gy, z=gz,
    mode='markers',
    marker=dict(size=3, color='rgba(255,255,255,0.05)'),
    hoverinfo='none',
    showlegend=False
)

# ----------------------------------------
# 4) Frames de animação
# ----------------------------------------
frames = []
iterations = sorted(df['iteration'].unique())

for it in iterations:
    sub = df[df['iteration'] == it]
    grouped = sub.groupby(['x', 'y', 'z'], as_index=False)['signal'].mean()

    trace = go.Scatter3d(
        x=grouped['x'],
        y=grouped['y'],
        z=grouped['z'],
        mode='markers',
        marker=dict(
            size=np.interp(grouped['signal'], [grouped['signal'].min(), grouped['signal'].max()], [7, 22]),
            color=grouped['signal'],
            cmin=grouped['signal'].min(),
            cmax=grouped['signal'].max(),
            # Paleta luminosa, de azul para amarelo
            colorscale='Jet',
            opacity=0.95,
            colorbar=dict(title='Signal', len=0.6, x=1.05)
        ),
        hovertemplate=('x: %{x}<br>y: %{y}<br>z: %{z}'
                       '<br>signal: %{marker.color:.3f}<extra></extra>'),
        name=f'iter {it}'
    )

    frames.append(go.Frame(
        data=[bg_trace, trace],
        name=str(it),  # mantém numérico para ordenação correta
        layout=go.Layout(title_text=f'Iteration: {it}')
    ))

# ----------------------------------------
# 5) Frame inicial e slider
# ----------------------------------------
initial_iter = iterations[0]
init_sub = df[df['iteration'] == initial_iter].groupby(['x', 'y', 'z'], as_index=False)['signal'].mean()

init_trace = go.Scatter3d(
    x=init_sub['x'],
    y=init_sub['y'],
    z=init_sub['z'],
    mode='markers',
    marker=dict(
        size=np.interp(init_sub['signal'], [sig_min, sig_max], [7, 22]),
        color=init_sub['signal'],
        cmin=sig_min,
        cmax=sig_max,
        colorscale='Jet',
        opacity=0.95
    ),
    name=f'iter {initial_iter}'
)

steps = []
for it in iterations:
    step = dict(
        method="animate",
        args=[[it],
              dict(mode="immediate",
                   frame=dict(duration=300, redraw=True),
                   transition=dict(duration=0))],
        label=str(it)
    )
    steps.append(step)

sliders = [dict(active=0, pad={"t": 50},
                steps=steps,
                currentvalue={"prefix": "Iteration: "})]

# ----------------------------------------
# 6) Layout dark
# ----------------------------------------
updatemenus = [
    dict(type="buttons",
         showactive=False,
         y=1.05,
         x=0.0,
         xanchor="left",
         yanchor="bottom",
         buttons=[
             dict(label="Play",
                  method="animate",
                  args=[None, dict(frame=dict(duration=400, redraw=True),
                                   transition=dict(duration=0),
                                   fromcurrent=True,
                                   mode='immediate')]),
             dict(label="Pause",
                  method="animate",
                  args=[[None], dict(frame=dict(duration=0, redraw=False),
                                     mode="immediate",
                                     transition=dict(duration=0))])
         ])
]

layout = go.Layout(
    width=900, height=700,
    scene=dict(
        xaxis=dict(title='X', color='white', backgroundcolor='black', gridcolor='#333'),
        yaxis=dict(title='Y', color='white', backgroundcolor='black', gridcolor='#333'),
        zaxis=dict(title='Z', color='white', backgroundcolor='black', gridcolor='#333'),
        bgcolor='black'
    ),
    paper_bgcolor='black',
    plot_bgcolor='black',
    font=dict(color='white'),
    updatemenus=updatemenus,
    sliders=sliders,
    margin=dict(l=0, r=200, b=0, t=50),
    title=f"Neural Activity"
)

fig = go.Figure(
    data=[bg_trace, init_trace],
    layout=layout,
    frames=frames
)

fig.write_html("neuron_activity.html", include_plotlyjs='cdn', auto_open=True)

###########################################################################

import pandas as pd
import plotly.graph_objects as go

with open('num_connections_history.txt', 'r') as f: 
    lines = f.readlines() 
    lines = [line.split() for line in lines] 
    df = pd.DataFrame(lines, columns = ['name', 'iteration', 'num_connections']) 
    df['iteration'] = df['iteration'].astype(int) 
    df['num_connections'] = df['num_connections'].astype(int)
    df = df.drop_duplicates(subset=['name', 'iteration'])

# Criar figura
fig = go.Figure()

# Adicionar uma linha para cada blob
for blob_name in df['name'].unique():
    df_blob = df[df['name'] == blob_name]
    fig.add_trace(go.Scatter(
        x=df_blob['iteration'],
        y=df_blob['num_connections'],
        mode='lines+markers',
        name=blob_name
    ))

# Layout do gráfico
fig.update_layout(
    title="Number of neuronal connections for each blob",
    xaxis_title="Iteration",
    yaxis_title="Number of connections",
    template="plotly_dark"  # opcional: deixa fundo escuro
)

# Salvar em HTML e abrir automaticamente
fig.write_html("blob_series.html", auto_open=True)

