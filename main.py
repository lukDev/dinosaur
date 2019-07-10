import pyglet
import metrics
from data import Data
import pyglet.window.key
from controller import Controller


window = pyglet.window.Window(width=Data.field_size_x, height=Data.field_size_y, resizable=True)
gen_label = pyglet.text.Label("Gen: ", x=10, y=30)
gen_label.color = (0, 0, 0, 255)
alive_players_label = pyglet.text.Label("Alive Players: 0", x=10, y=10)
alive_players_label.color = (0, 0, 0, 255)
fitness_label = pyglet.text.Label("0", x=10, y=Data.field_size_y - 25)
fitness_label.color = (0, 0, 0, 255)
Controller.reset_level()


# --- Events ---
@window.event
def on_draw():
    window.clear()
    pyglet.gl.glClearColor(1., 1., 1., 1.)

    pyglet.gl.glLineWidth(5)
    pyglet.gl.glColor3f(0., 0., 0.)
    pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ("v2f", [0, Data.baseline, Data.field_size_x, Data.baseline]))

    for player in Data.players:
        if player.alive:
            player.render()

    for obstacle in Data.obstacles:
        obstacle.render()

    gen_label.draw()
    alive_players_label.draw()
    fitness_label.draw()


# --- Functions ---
def update(dt):
    Data.overall_time += dt
    alive_count = Controller.update(dt)

    gen_label.text = "Gen: " + str(Data.gen)
    alive_players_label.text = "Alive Players: %d" % alive_count
    score = int(Data.overall_time * 10)
    fitness_label.text = "Score: %d" % score
    on_draw()

    if alive_count == 0:
        next_gen()


def next_gen():
    pyglet.clock.unschedule(update)
    Controller.reset_level()
    Data.overall_time = 0.

    if len(Data.players) == 0:
        metrics.init_starter_players()
    else:
        metrics.evolve([(p.network, p.fitness) for p in Data.players])

    Data.gen += 1

    pyglet.clock.schedule_interval(update, 1. / 60.)


next_gen()
pyglet.app.run()
