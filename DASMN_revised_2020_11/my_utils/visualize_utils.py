# import visdom
# vis = visdom.Visdom(env='yancy_env')


# suggested function
class Visualize_v2:
    def __init__(self, vis):
        self.vis = vis

    def plot(self, data, label, counter, scenario):
        assert len(data) == len(label)
        if len(data) == 1:
            self.vis.line(Y=[data[0]], X=[counter],
                          update=None if counter == 0 else 'append', win=scenario,
                          opts=dict(legend=[label[0]], title=scenario))
        elif len(data) == 2:
            self.vis.line(Y=[[data[0], data[1]]], X=[counter],
                          update=None if counter == 0 else 'append', win=scenario,
                          opts=dict(legend=[label[0], label[1]], title=scenario))
        elif len(data) == 3:
            self.vis.line(Y=[[data[0], data[1], data[2]]], X=[counter],
                          update=None if counter == 0 else 'append', win=scenario,
                          opts=dict(legend=[label[0], label[1], label[2]], title=scenario))

        elif len(data) == 4:
            self.vis.line(Y=[[data[0], data[1], data[2], data[3]]], X=[counter],
                          update=None if counter == 0 else 'append', win=scenario,
                          opts=dict(legend=[label[0], label[1], label[2], label[3]], title=scenario))


if __name__ == "__main__":
    a = Visualize_v2(vis=None)
    pass
