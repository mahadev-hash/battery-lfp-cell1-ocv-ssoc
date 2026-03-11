import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ================= USER INPUT =================
CSV_PATH = "LFP-1.csv"
DV = 0.005
ICA_RANGE = [-200, 200]

# ================= FUNCTIONS =================
def interpolate_and_dqdv(V, Q, dv):

    idx = np.argsort(V)
    V = V[idx]
    Q = Q[idx]

    Vn = np.arange(V.min(), V.max(), dv)
    Qi = np.interp(Vn, V, Q)

    dqdv = np.gradient(Qi, Vn)

    return Vn, dqdv


def capacity_to_soc(Q):
    """Convert capacity to SOC"""
    Qmin = np.min(Q)
    Qmax = np.max(Q)

    soc = (Q - Qmin) / (Qmax - Qmin) * 100

    return soc


# ================= LOAD DATA =================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

n_cells = len(df.columns) // 4
cell_names = [f"Cell-{i+1}" for i in range(n_cells)]
colors = px.colors.qualitative.Dark24

# ================= DASH APP =================
app = Dash(__name__)

app.layout = html.Div([

    html.H2("Battery OCV / ICA Dashboard", style={"textAlign": "center"}),

    html.Div([

        html.Div([
            html.Label("Cell A"),
            dcc.Dropdown(
                options=[{"label": c, "value": i} for i, c in enumerate(cell_names)],
                value=0,
                id="cell-a"
            )
        ], style={"width": "30%"}),

        html.Div([
            html.Label("Cell B"),
            dcc.Dropdown(
                options=[{"label": c, "value": i} for i, c in enumerate(cell_names)],
                value=1,
                id="cell-b"
            )
        ], style={"width": "30%"}),

        html.Div([
            html.Label("Mode"),
            dcc.Dropdown(
                options=[
                    {"label": "Single Cell", "value": "single"},
                    {"label": "Compare", "value": "compare"}
                ],
                value="single",
                id="mode"
            )
        ], style={"width": "30%"})

    ], style={"display": "flex", "gap": "20px"}),

    dcc.Graph(id="battery-graph", style={"height": "650px"})

], style={"margin": "20px"})


# ================= CALLBACK =================
@app.callback(
    Output("battery-graph", "figure"),
    Input("cell-a", "value"),
    Input("cell-b", "value"),
    Input("mode", "value")
)
def update_graph(cell_a, cell_b, mode):

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=["OCV (Voltage vs SOC)", "ICA (dQ/dV vs Voltage)"]
    )

    def get_cell_data(i):

        c = 4*i

        Vc = df.iloc[:, c].dropna().to_numpy()
        Qc = df.iloc[:, c+1].dropna().to_numpy()

        Vd = df.iloc[:, c+2].dropna().to_numpy()
        Qd = df.iloc[:, c+3].dropna().to_numpy()

        # SOC calculation
        SOCc = capacity_to_soc(Qc)
        SOCd = capacity_to_soc(Qd)

        # ICA
        Vci, dQci = interpolate_and_dqdv(Vc, Qc, DV)
        Vdi, dQdi = interpolate_and_dqdv(Vd, Qd, DV)

        return SOCc, Vc, SOCd, Vd, dQci, Vci, dQdi, Vdi


    # background curves
    for i in range(n_cells):

        SOCc, Vc, SOCd, Vd, dQci, Vci, dQdi, Vdi = get_cell_data(i)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=SOCc,
            y=Vc,
            opacity=0.2,
            line=dict(color=color),
            showlegend=False
        ),1,1)

        fig.add_trace(go.Scatter(
            x=SOCd,
            y=Vd,
            opacity=0.2,
            line=dict(color=color),
            showlegend=False
        ),1,1)

        fig.add_trace(go.Scatter(
            x=dQci,
            y=Vci,
            opacity=0.2,
            line=dict(color=color),
            showlegend=False
        ),1,2)

        fig.add_trace(go.Scatter(
            x=dQdi,
            y=Vdi,
            opacity=0.2,
            line=dict(color=color),
            showlegend=False
        ),1,2)


    def highlight_cell(i,label):

        SOCc, Vc, SOCd, Vd, dQci, Vci, dQdi, Vdi = get_cell_data(i)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=SOCc,
            y=Vc,
            name=f"{label} Charge",
            line=dict(width=3,color=color)
        ),1,1)

        fig.add_trace(go.Scatter(
            x=SOCd,
            y=Vd,
            name=f"{label} Discharge",
            line=dict(width=3,color=color)
        ),1,1)

        fig.add_trace(go.Scatter(
            x=dQci,
            y=Vci,
            line=dict(width=3,color=color),
            showlegend=False
        ),1,2)

        fig.add_trace(go.Scatter(
            x=dQdi,
            y=Vdi,
            line=dict(width=3,color=color),
            showlegend=False
        ),1,2)


    highlight_cell(cell_a, cell_names[cell_a])

    if mode=="compare" and cell_b!=cell_a:
        highlight_cell(cell_b, cell_names[cell_b])


    fig.update_layout(
        template="plotly_white",
        hovermode="closest"
    )

    fig.update_xaxes(title="SOC (%)",range=[0,100],row=1,col=1)
    fig.update_xaxes(title="dQ/dV (mAh/V)",range=ICA_RANGE,row=1,col=2)
    fig.update_yaxes(title="Voltage (V)")

    return fig


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)

