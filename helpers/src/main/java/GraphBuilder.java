import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

import java.util.UUID;

/**
 * Created by alse on 8/30/17.
 * Project: https://github.com/alseambusher/Tensorflow-Examples4j
 *
 * Api for graph operations are not available for java yet, so let us create a class for it
 */

class GraphBuilder {
    GraphBuilder(Graph g) {
        this.graph = g;
    }

    Output add(Output x, Output y) {
        return binaryOp("Add", x, y);
    }

    Output sub(Output x, Output y) {
        return binaryOp("Sub", x, y);
    }
    Output mul(Output x, Output y) {
        return binaryOp("Mul", x, y);
    }

    Output div(Output x, Output y) {
        return binaryOp("Div", x, y);
    }

    Output matmul(Output x, Output y) {
        return binaryOp("MatMul", x, y);
    }

    Output resizeBilinear(Output images, Output size) {
        return binaryOp("ResizeBilinear", images, size);
    }

    Output expandDims(Output input, Output dim) {
        return binaryOp("ExpandDims", input, dim);
    }

    Output cast(Output value, DataType dtype) {
        return graph.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
    }

    Output decodeJpeg(Output contents, long channels) {
        return graph.opBuilder("DecodeJpeg", "DecodeJpeg")
                .addInput(contents)
                .setAttr("channels", channels)
                .build()
                .output(0);
    }

    Output constant(String name, Object value) {
        try (Tensor t = Tensor.create(value)) {
            return graph.opBuilder("Const", name)
                    .setAttr("dtype", t.dataType())
                    .setAttr("value", t)
                    .build()
                    .output(0);
        }
    }

    Output placeholder(String name, DataType dtype) {
        return graph.opBuilder("Placeholder", name)
                .setAttr("dtype", dtype)
                .build()
                .output(0);
    }

    private Output binaryOp(String type, Output in1, Output in2) {
        return graph.opBuilder(type, UUID.randomUUID().toString()).addInput(in1).addInput(in2).build().output(0);
    }

    public Graph graph;
}
