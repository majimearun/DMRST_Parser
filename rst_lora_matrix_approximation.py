import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class RSTOutput:
    """
    Dataclass to store different RST matrix representations
    """
    rst_binary_without_labels: np.ndarray 
    # rst_binary_with_labels: np.ndarray
    rst_prob_without_labels: np.ndarray
    # rst_prob_with_labels: np.ndarray

class RSTMatrixProcessor:
    def __init__(self, num_relation_types: int):
        """
        Initialize the RST Matrix Processor
        
        Args:
            num_relation_types: Number of possible discourse relation types
        """
        self.num_relation_types = num_relation_types
    
    def create_rst_matrix(self, 
                         edus: List[str], 
                         rst_parser_output: List[Tuple[int, int, int, float]]) -> np.ndarray:
        """
        Create initial 3D RST matrix from parser output
        
        Args:
            edus: List of Elementary Discourse Units (EDUs)
            rst_parser_output: List of tuples (edu_i, edu_j, relation_k, probability)
                             where edu_i is nucleus, edu_j is satellite, 
                             relation_k is relation type index, probability is connection probability
        
        Returns:
            3D numpy array of shape (num_edus, num_edus, num_relation_types)
        """
        num_edus = len(edus)
        rst_matrix = np.zeros((num_edus, num_edus, self.num_relation_types))
        for edu_i, edu_j, rel_k, prob in rst_parser_output:
            if edu_i != edu_j:
                rst_matrix[edu_i, edu_j, rel_k] = prob
                
        return rst_matrix
    
    def compute_importance_index(self, rst_matrix: np.ndarray) -> np.ndarray:
        """
        Average and merge y-axis to compute importance index
        
        Args:
            rst_matrix: 3D RST matrix
        
        Returns:
            2D matrix of importance indices
        """
        return np.mean(rst_matrix, axis=1)
    
    def create_binary_matrix(self, prob_matrix: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert probabilistic matrix to binary matrix
        
        Args:
            prob_matrix: Matrix with probability values
            threshold: Threshold for binary conversion (default: 0.5)
        
        Returns:
            Binary matrix
        """
        return (prob_matrix >= threshold).astype(np.int32)
    
    def process_rst(self, 
                   edus: List[str], 
                   rst_parser_output: List[Tuple[int, int, int, float]]) -> RSTOutput:
        """
        Process RST parser output into four different matrix distributions
        
        Args:
            edus: List of Elementary Discourse Units
            rst_parser_output: Raw parser output as list of tuples
        
        Returns:
            RSTOutput object containing all four matrix representations
        """
        initial_matrix = self.create_rst_matrix(edus, rst_parser_output)
        importance_matrix = self.compute_importance_index(initial_matrix)
        rst_b_wo = self.create_binary_matrix(importance_matrix)
        # rst_b_w = self.create_binary_matrix(importance_matrix)
        rst_p_wo = importance_matrix
        # rst_p_w = importance_matrix
        
        self.rst_output = RSTOutput(
            rst_binary_without_labels=rst_b_wo,
            # rst_binary_with_labels=rst_b_w,
            rst_prob_without_labels=rst_p_wo,
            # rst_prob_with_labels=rst_p_w
        )
        
    def create_retrieval_map(self):
        self.binary_retrieval_map = {}
        for i in range(self.rst_output.rst_binary_without_labels.shape[0]):
            for j in range(self.rst_output.rst_binary_without_labels.shape[1]):
                self.binary_retrieval_map[(i+1, j+1)] = self.rst_output.rst_binary_without_labels[i, j]
                
        self.prob_retrieval_map = {}
        for i in range(self.rst_output.rst_prob_without_labels.shape[0]):
            for j in range(self.rst_output.rst_prob_without_labels.shape[1]):
                self.prob_retrieval_map[(i+1, j+1)] = self.rst_output.rst_prob_without_labels[i, j]
                
                
def get_key_based_on_index(index: int, retrieval_map: Dict[Tuple[int, int], float]):
    result = []
    for key, value in retrieval_map.items():
        if key[0] == index:
            result.append(value)
    return np.array(result)
        

def main():
    edus = [
        "A",
        "B",
        "C"
    ]
    
    parser_output = [
        (0, 1, 0, 0.8), 
        (1, 2, 2, 0.6)
    ]
    
    processor = RSTMatrixProcessor(num_relation_types=3)
    processor.process_rst(edus, parser_output)
    
    # print("Binary matrix without labels shape:", processor.rst_output.rst_binary_without_labels.shape)
    # print(processor.rst_output.rst_binary_without_labels)
    # print("Binary matrix with labels shape:", result.rst_binary_with_labels.shape)
    # print(result.rst_binary_with_labels)
    # print("Probabilistic matrix without labels shape:", processor.rst_output.rst_prob_without_labels.shape)
    # print(processor.rst_output.rst_prob_without_labels)
    # print("Probabilistic matrix with labels shape:", result.rst_prob_with_labels.shape)
    # print(result.rst_prob_with_labels)
    
    processor.create_retrieval_map()
    print("Binary retrieval map:")
    print(processor.binary_retrieval_map)
    print("Probabilistic retrieval map:")
    print(processor.prob_retrieval_map)
    
    print("Get key based on index 1:")
    print(get_key_based_on_index(1, processor.prob_retrieval_map))
    
    

main()
