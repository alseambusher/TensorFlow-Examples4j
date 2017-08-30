import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * Created by alse on 8/30/17.
 * Project: https://github.com/alseambusher/Tensorflow-Examples4j
 */
public class HelloWorld {
    public static void main (String args[]) {
        Graph g = new Graph();
        String message = "Hello, Tensorflow";
        Tensor hello = Tensor.create(message.getBytes());
        g.opBuilder("Const", "hello")
                .setAttr("dtype", hello.dataType())
                .setAttr("value", hello)
                .build()
                .output(0);
        Session s =  new Session(g);
        byte [] output = s.runner().fetch("hello").run().get(0).bytesValue();
        System.out.println(new String(output));
    }
}
