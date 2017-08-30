import org.tensorflow.*;

import java.util.UUID;

/**
 * Created by alse on 8/30/17.
 * Project: https://github.com/alseambusher/Tensorflow-Examples4j
 *
 * This covers basic operations such as
 * 1. Creating constants, placeholders
 * 2. Add, Sub, Mul, Div
 * 3. Matrix multiplication
 */

public class BasicOperations {
    public static void main(String args[]) {
        // Create a new graph
        GraphBuilder g = new GraphBuilder(new Graph());

        // Basic constant operations
        Output a = g.constant("a", 1);
        Output b = g.constant("b", 3);

        // add sub mul and div operations
        Output c = g.add(a, b);
        Output d = g.sub(a, b);
        Output e = g.mul(a, b);
        Output f = g.div(a, b);

        // create a new session and run the operations
        Session s =  new Session(g.graph);
        System.out.println(s.runner().fetch(c.op().name()).run().get(0).intValue());
        System.out.println(s.runner().fetch(d.op().name()).run().get(0).intValue());
        System.out.println(s.runner().fetch(e.op().name()).run().get(0).intValue());
        System.out.println(s.runner().fetch(f.op().name()).run().get(0).intValue());

        // matrix multiplication
        float[][] X = new float[][]{new float[]{1,2,3}};
        float[][] Y = new float[][]{new float[]{4}, new float[]{5}, new float[]{6}};

        Output h = g.constant("h", X);
        Output i = g.constant("i", Y);

        Output j = g.matmul(i, h);
        System.out.println(s.runner().fetch(j.op().name()).run().get(0).toString());


        // using placeholder
        Output k = g.placeholder("k", DataType.INT32);
        Output l = g.add(c, k);

        // create a tensor to store the value for k
        Tensor k_value = Tensor.create(10);

        // now feed the value for the placeholder and compute l
        System.out.println(s.runner().feed("k", k_value).fetch(l.op().name()).run().get(0).intValue());

    }

    // Api for graph operations are not available for java yet, so let us create a class for it
    private static class GraphBuilder {
        public Graph graph;

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
    }
}
