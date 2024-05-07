from typing import Tuple
from torch import Tensor, from_numpy, float32
from sklearn.ensemble import RandomForestClassifier
import pickle
from .abstract import NonIterativeModel



class RandomForest(NonIterativeModel):
    """
    Not PyTorch model
    """
    def __init__(self, fdim:int, n_classes:int,
                 max_depth:int, training:bool=False):
        self.max_depth = max_depth
        self.model = RandomForestClassifier(max_depth=max_depth, 
                                            random_state=0)

    def to_device(self, device:str):
        """ Not implemented for this model
        """
        pass
        
    def fit(self, x:Tensor, y:Tensor):
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        self.model.fit(x, y)
        
    def predict(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (B, F): batch of input features
        Returns:
            tuple
              None: output logits (loss backprop is not supported)
              (B, C): output probs (output of softmax)
        """
        x = x.cpu().numpy()
        probs = self.model.predict_proba(x)
        probs = from_numpy(probs).to(float32)
        logits = None
        return logits, probs

    def save(self, weights_path:str):
        with open(weights_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, weights_path:str):
        with open(weights_path, 'rb') as f:
            model = pickle.load(f)
        # check match parameters
        try:
            assert model.max_depth == self.max_depth, "max_depth"
        except AssertionError as err:
            raise Exception(f"Mismatch parameters: {err}")
        self = model

    
if __name__ == '__main__':
    pass    