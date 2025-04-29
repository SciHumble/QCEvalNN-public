from qc_eval.evaluation import Evaluation

"""
Final Testing and Evaluation:
    Classical: 4, 8, 16, 32 input
    Quantum: 4, 8, 16 Qubits (with dense and normal qubit encoding)
    
    For QCNN all configurations should get tested 5 times.
    
    For CCNN all configuration that are in the range or the closest to the
    results of QCNN get 5 times trained and tested to find the spread.
"""

if __name__ == "__main__":
    evaluator = Evaluation("mnist")
    evaluator.start(ccnn=False, qcnn=True)
