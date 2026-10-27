from nipype import Node
from nipype.interfaces.fsl.fix import Classifier


def classifier_node(trained_wts_file, thresh):
    node = Node(
        Classifier(command="fix -c"),
        name="classifier",
    )
    node.inputs.trained_wts_file = trained_wts_file
    node.inputs.thresh = thresh
    return node
