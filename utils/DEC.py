import keras
from keras import layers

from sklearn.cluster import KMeans
import keras.backend as K
from keras.engine.topology import Layer



class ClusteringLayer(Layer):

      
    def __init__(self, n_clusters=10, weights=None, alpha=1.0, **kwargs):

        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights


    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1] # 클러스터 개수
        
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform')
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
        


    def call(self, inputs, **kwargs):
      
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q


class DEC():
    
    def __init__(self):
        self.n_clusters = 10
    
    def target_distribution(self,q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    
    def autoencoder(self):
        
        x = keras.Input(shape=(784,))
        encoder = layers.Dense(500, activation='relu')(x)
        encoder = layers.Dense(500, activation='relu')(encoder)
        encoder = layers.Dense(2000, activation='relu')(encoder)
        encoder = layers.Dense(10, activation='relu')(encoder)

    
        decoder = layers.Dense(2000, activation='relu')(encoder)
        decoder = layers.Dense(500, activation='relu')(decoder)
        decoder = layers.Dense(500, activation='relu')(decoder)
        decoder = layers.Dense(784, activation='relu')(decoder)
    
        
        return keras.Model(x, decoder), keras.Model(x, encoder)
        
    
    def run(self, x):
        
        autoencoder, encoder = self.autoencoder()       
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(x, x, batch_size=256, epochs=1)
        z = encoder.predict(x)
        
        
        clustering_layer = ClusteringLayer(self.n_clusters,name='clustering')(encoder.output) #(None, 10)
        model = keras.Model(encoder.input, clustering_layer) 
        model.compile(optimizer='adam', loss='kld')
        
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        a=kmeans.fit_predict(z)
        print(a)
        model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        
        
        
        for ite in range(1):
            
            q = model.predict(x)
            p = self.target_distribution(q)
            print(z.shape)
            print(p.shape)
            loss = model.train_on_batch(x, p)
        
        
        res = model.predict(x)
        return res.argmax(1)
    




