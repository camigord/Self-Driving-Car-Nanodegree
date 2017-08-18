import numpy as np

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        self.k = 5
        self.last_fits = []
        self.radius_of_curvature = []
        self.line_base_pos = []
        for i in range(self.k):
            self.last_fits.append(None)
            self.radius_of_curvature.append(None)
            self.line_base_pos.append(None)

        self.temp_fit = [np.array([False])]
        self.init = True
        self.temp_counter = 0

    def add_stats(self, detected, fit, curvature, distance):
        self.detected = detected
        if self.detected:
            self.current_fit = fit
            self.last_fits.append(fit)
            self.radius_of_curvature.append(curvature)
            self.line_base_pos.append(distance)
            if self.init:
                self.init = False
                self.temp_fit = fit
            else:
                self.temp_fit = self.temp_fit*0.8 + fit*0.2

        else:
            self.last_fits.append(None)
            self.radius_of_curvature.append(None)
            self.line_base_pos.append(None)
            self.temp_counter += 1
            if self.temp_counter == 5:
                self.temp_counter = 0
                self.init = True

        self.last_fits = self.last_fits[1:]
        self.radius_of_curvature = self.radius_of_curvature[1:]
        self.line_base_pos = self.line_base_pos[1:]


    def get_best_fit(self):
        temp = np.zeros([3])
        curv = 0
        dist = 0
        count = 0
        for i in range(self.k):
            if self.last_fits[i] is not None:
                temp += self.last_fits[i]
                curv += self.radius_of_curvature[i]
                dist += self.line_base_pos[i]
                count += 1

        '''
        if count == 0:
            return None
        else:
            return (temp/float(count), curv/float(count), dist/float(count))
        '''
        if self.init:
            return None
        else:
            return (self.temp_fit, curv/float(count), dist/float(count))
