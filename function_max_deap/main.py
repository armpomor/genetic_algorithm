import plotly.graph_objs as go
import numpy as np

from ga import FuncMax


ga = FuncMax()

x = [i[0] for i in ga.population]
y = [i[1] for i in ga.population]

max_fitness_values, mean_fitness_values = ga.run().select('max', 'avg')

fig = go.Figure()
fig.update_layout(title=dict(text='Genetic algorithm',
                             font=dict(size=20)), margin=dict(l=0, t=30, b=0, r=0))
fig.add_trace(go.Scatter(y=max_fitness_values, mode='lines+markers', name='Max values',
                         marker=dict(size=5, color='red'), line=dict(width=2, color='gray')))
fig.add_trace(go.Scatter(y=mean_fitness_values, mode='lines+markers', name='Mean values',
                         marker=dict(size=5, color='blueviolet'), line=dict(width=2, color='chocolate')))
fig.update_xaxes(title='Generation')
fig.update_yaxes(title='Values of the function z = sin(x) * 100 / (x ** 2 + 2 * y ** 2 + 50)')

fig.show()


x_val, y_val = np.meshgrid(x, y)

z = ga.one_fitness([x_val, y_val])[0]

fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Viridis")])
fig.update_layout(title='z = sin(x) * 100 / (x ** 2 + 2 * y ** 2 + 50)',
                  width=1500, height=800,
                  margin=dict(l=0, t=30, b=0, r=0))
fig.show()


