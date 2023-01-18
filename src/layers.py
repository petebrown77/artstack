import numpy as np

class Layer(np.ndarray):

    def __json__(self):
        return {
            "name" : self.__class__.__name__,
            "params" : self.params
        }

    @classmethod
    def from_json(self):
        raise NotImplementedError("method from_json has not been implemented")
    
    def __str__(self):
        return "{name}\narray: shape {array}".format(name=self.__class__.__name__, array=self.shape)
    
    def __init__(self, array, **kwargs):
        pass

    def __new__(cls, array, **kwargs):
        print("Creating new class")
        #    continue
        if kwargs:
            print("Kwargs found: {}".format(kwargs))
            cls.params = kwargs
        else:
            cls.params = cls.create_values(cls, cls.rand_param(cls))
            #    pass
            #cls.params = cls.create_values(cls.rand_param(cls))
        print("Params: ",cls.params)
            
        #TODO: apply self function here
        obj = cls.apply(cls, np.asarray(array), **cls.params).view(cls)
        return obj
    
    def randomizer(self, list_val):
        if isinstance(list_val[0], int):
            return random.randint(*list_val)
        elif isinstance(list_val[0], float):
            return random.uniform(*list_val)
            
    def create_values(self, values):
        return {k: self.randomizer(self, v) for k,v in values.items()}
    
    def update_parameters(self, values):
        """
            Update parameters of layer, and update underlying array.
            Parameters: values -> dict, dictionary of values to update, and the key of the value to update to.
            Returns: Class Layer -> return updated array value
        """
        for k, v in values.items():
            self.params[k] = v
            
        self = self.apply(np.asarray(self), **self.params).view(self.__class__)
        return self
    
    def rand_param(self):
        raise NotImplementedError("rand_param() must be implemented in new Layer class {}".format(self.__class__.__name__))
        
    def apply(self, array, **kwargs):
        raise NotImplementedError("apply() must be implemented in {}".format(self.__class__.__name__))




class PerlinNoiseLayer(Layer):
    def rand_param(self):
        return {
            "scale": [1, 1],
            "octaves": [2, 10],
            "persistence": [0.0, 0.1],
            "lacunarity": [3, 9],
            "base": [1, 300]
        }

    def apply(self, array, octaves=8, scale=1, persistence=0.8, lacunarity=3, base=2):

        x_idx = np.linspace(0, 1, array.shape[0])
        y_idx = np.linspace(0, 1, array.shape[1])

        world_x, world_y = np.meshgrid(x_idx, y_idx)

        return np.vectorize(noise.pnoise2)(world_x/scale,
                                           world_y/scale,
                                           octaves=octaves,
                                           persistence=persistence,
                                           lacunarity=lacunarity,
                                           base=base)



class NormalizeLayer(Layer):
    
    def rand_param(self):
        return {}
    
    def apply(self, array):
        return (array - np.min(array))/ (np.max(array) - np.min(array))


class GaussianFilter(Layer):
    def rand_param(self):
        return {
            "sigma": [2, 15]
        }
    def apply(self, array, sigma=75):
        print("Sigma value: {}".format(sigma))
        return gaussian_filter(array, sigma)


class ColorMapTransform(Layer):
    
    def randomize_params(self):
        #can't do randomize yet
        return {
            colormap: []
        }
    
    def apply(self, array, colormap=map):
        
        color_array = np.zeros((array.shape[0], array.shape[1], 3))
        
        #using colormap and np.where,
        for value, color in self.params["colormap"]:
            subset = np.where(array > value)
            color_array[subset] = color
            
        return color_array


