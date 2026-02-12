import numpy as np


class Yoshimura:

    def __init__(self, n, beta, randomize=False):
        self.n = n
        self.beta = beta
        self.d = np.tan(self.beta)
        self.w = 0.5 * np.tan(self.beta)
        self.l = 0.5
        self.b = 0.5 / np.cos(self.beta)

        self.r = self.l / (2 * np.sin(np.pi / (2 * self.n)))
        self.nodes = []
        self.base_nodes = np.array([
            [self.r * np.sin(np.pi / self.n * i),
             -self.r * np.cos(np.pi / self.n * i),
             0]
            for i in range(2 * self.n)
        ])
        self.mid_nodes = np.array([
            [self.r * np.sin(np.pi / self.n * i),
             -self.r * np.cos(np.pi / self.n * i),
             self.d / 2]
            for i in range(2 * self.n)
        ])

        self.top_nodes = np.array([
            [self.r * np.sin(np.pi / self.n * i),
             -self.r * np.cos(np.pi / self.n * i),
             self.d]
            for i in range(2 * self.n)
        ])

        self.nodes.extend(self.base_nodes)  # indices 0 to 2n-1
        self.nodes.extend(self.mid_nodes)  # indices 2n to 4n-1
        self.nodes.extend(self.top_nodes)  # indices 4n to 6n-1

        if randomize:
            # start nodes with random coordinates of size (6*n, 3) uniformly distributed along origin
            self.nodes += (1 - 2 * np.random.random((6 * self.n, 3))) * 0.01

        self.bars = []
        self.hinges = []
        self.faces = []

    def base_idx(self, i):
        """Get index in base layer (0 to 2n-1)"""
        return i % (2 * self.n)

    def mid_idx(self, i):
        """Get index in mid layer (2n to 4n-1)"""
        return 2 * self.n + i % (2 * self.n)

    def top_idx(self, i):
        """Get index in top layer (4n to 6n-1)"""
        return 4 * self.n + i % (2 * self.n)

    def get_geometry(self):
        for i in range(2 * self.n):
            j = i + 1
            k = i - 1

            self.bars.append((self.base_idx(i), self.base_idx(j), self.l))
            self.bars.append((self.mid_idx(i), self.mid_idx(j), self.l))
            self.bars.append((self.top_idx(i), self.top_idx(j), self.l))
            self.bars.append((self.base_idx(i), self.mid_idx(i), self.w))
            self.bars.append((self.mid_idx(i), self.top_idx(i), self.w))

            if i % 2 == 0:
                self.bars.append((self.base_idx(i), self.mid_idx(j), self.b))
                self.bars.append((self.base_idx(i), self.mid_idx(k), self.b))
                self.bars.append((self.top_idx(i), self.mid_idx(j), self.b))
                self.bars.append((self.top_idx(i), self.mid_idx(k), self.b))

                self.faces.append((self.base_idx(i), self.mid_idx(i), self.mid_idx(j)))
                self.faces.append((self.base_idx(i), self.mid_idx(i), self.mid_idx(k)))
                self.faces.append((self.base_idx(i), self.base_idx(j), self.mid_idx(j)))
                self.faces.append((self.base_idx(i), self.base_idx(k), self.mid_idx(k)))
                self.faces.append((self.top_idx(i), self.mid_idx(i), self.mid_idx(j)))
                self.faces.append((self.top_idx(i), self.mid_idx(i), self.mid_idx(k)))
                self.faces.append((self.top_idx(i), self.top_idx(j), self.mid_idx(j)))
                self.faces.append((self.top_idx(i), self.top_idx(k), self.mid_idx(k)))

                self.hinges.append(
                    (self.base_idx(i), self.mid_idx(j), self.base_idx(j), self.mid_idx(i), np.pi, 'fold'))
                self.hinges.append((self.top_idx(i), self.mid_idx(j), self.top_idx(j), self.mid_idx(i), np.pi, 'fold'))
                self.hinges.append((self.mid_idx(i), self.mid_idx(j), self.top_idx(i), self.base_idx(i), np.pi, 'fold'))
                self.hinges.append((self.mid_idx(i), self.mid_idx(k), self.top_idx(i), self.base_idx(i), np.pi, 'fold'))
                self.hinges.append(
                    (self.mid_idx(i), self.base_idx(k), self.base_idx(i), self.mid_idx(k), np.pi, 'fold'))
                self.hinges.append((self.mid_idx(i), self.top_idx(k), self.top_idx(i), self.mid_idx(k), np.pi, 'fold'))
                self.hinges.append((self.top_idx(i), self.mid_idx(i), self.mid_idx(j), self.mid_idx(k), np.pi, 'facet'))
                self.hinges.append(
                    (self.base_idx(i), self.mid_idx(i), self.mid_idx(j), self.mid_idx(k), np.pi, 'facet'))
            else:
                self.hinges.append(
                    (self.base_idx(i), self.mid_idx(i), self.base_idx(j), self.base_idx(k), np.pi, 'facet'))
                self.hinges.append((self.top_idx(i), self.mid_idx(i), self.top_idx(j), self.top_idx(k), np.pi, 'facet'))

        return self.nodes, self.bars, self.hinges, self.faces, None
