import plotly.graph_objs as go
import numpy as np

from gen_alg import Func_max


ga = Func_max()

x = ga.x
y = ga.y

ga.run()

z_mean = ga.average_values_z

fig = go.Figure()
fig.update_layout(title=dict(text='Genetic algorithm',
                  font=dict(size=20)), margin=dict(l=0, t=30, b=0, r=0))
fig.add_trace(go.Scatter(y=z_mean, mode='lines+markers',
              marker=dict(size=5, color='red'), line=dict(width=3, color='gray')))
fig.update_xaxes(title='Generation')
fig.update_yaxes(title='Mean values of the function z = sin(x) * 100 / (x ** 2 + 2 * y ** 2 + 50)')

fig.show()


x_val, y_val = np.meshgrid(x, y)

z = ga.fitness(x_val, y_val)

fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Viridis")])
fig.update_layout(title='z = sin(x) * 100 / (x ** 2 + 2 * y ** 2 + 50)',
                  width=1500, height=800,
                  margin=dict(l=0, t=30, b=0, r=0))
fig.show()
