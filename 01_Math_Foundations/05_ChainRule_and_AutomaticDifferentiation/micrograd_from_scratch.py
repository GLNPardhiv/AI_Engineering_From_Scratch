import math


class Value:
    """
    A scalar wrapper that constructs a directed acyclic graph (DAG) for automatic differentiation.

    This class encapsulates a individual numerical value and maintains explicit references
    to its parent nodes. By tracking these connections alongside an operation label, it provides
    the underlying data structure required to execute backpropagation and compute gradients
    via the calculus chain rule.

    Attributes:
        data (float): The forward-pass numerical value computed and stored in this node.
        grad (float): The accumulated derivative of the final output node with respect to
            this specific node. Defaults to 0.0.
        _backward (callable): A private function responsible for computing and propagating
            gradients back to the direct inputs of this node. Defaults to a no-op lambda.
        _prev (set[Value]): A unique set tracking the direct parent nodes that generated
            this node.
        _op (str): A string label indicating the mathematical operator used to produce
            this node.
    """
    def __init__(self, data, children=(), op=''):
        """
        Initializes a new node inside the computational graph.

        Args:
            data (float or int): The core numerical value you want to store.
            children (tuple, optional): The original Value objects that produced this new one.
                Defaults to an empty tuple.
            op (str, optional): The character symbol for the operation that created this node.
                Defaults to an empty string.

        Returns:
            None: Initializes the object state directly.
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        """
        Returns a clean string representation of the object for debugging.

        Returns:
            str: A formatted label showing the node's current data and gradient
                rounded to 4 decimal places.
        """
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        """
        Performs element-wise addition between this node and another value.

        This method executes the forward addition step, registers both inputs as
        parent nodes in the computational graph, and defines the backward closure
        to propagate gradients using the addition rule of calculus.

        Args:
            other (Value or float or int): The right-hand value to add. If it is
                a raw number, it is automatically wrapped in a Value node.

        Returns:
            Value: A new node containing the sum, linked to its parent inputs,
                and armed with its specific gradient routing function.
        """
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            """Propagates the output gradient directly to both parent inputs."""
            self.grad += out.grad
            other.grad += out.grad

        # Attach the backward function to the output node
        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Performs element-wise multiplication between this node and another value.

        This method executes the forward multiplication step, registers both inputs as
        parent nodes in the computational graph, and defines the backward closure
        to propagate gradients using the product rule of calculus.

        Args:
            other (Value or float or int): The right-hand value to multiply. If it is
                a raw number, it is automatically wrapped in a Value node.

        Returns:
            Value: A new node containing the product, linked to its parent inputs,
                and armed with its specific gradient routing function.
        """
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            """Propagates gradients by scaling the output gradient by the opposite input's data."""
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """
        Applies the Rectified Linear Unit (ReLU) activation function to this node.

        In the forward pass, this caps the value at zero (if the number is negative,
        it becomes 0; if it is positive, it stays the same). In the backward pass,
        it acts as a gatekeeper for gradients: it lets the gradient pass through
        completely if the output was positive, but blocks it completely (sets it to 0)
        if the output was capped at zero.

        Returns:
            Value: A new node containing the rectified value, linked back to this
                node as its parent, and armed with its gradient gating function.
        """
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            """Gates the incoming gradient based on whether the node was active (> 0)."""
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        """
        Negates the numerical sign of this node (equivalent to self * -1).

        Returns:
            Value: A new node representing the negative version of this node's data.
        """
        return self * -1

    def __sub__(self, other):
        """
        Performs subtraction by rewriting the operation as adding a negative value (self + (-other)).

        Args:
            other (Value or float or int): The right-hand value to subtract.

        Returns:
            Value: A new node containing the difference.
        """
        return self + (-other)

    def __radd__(self, other):
        """
        Handles fallback addition when a raw number is on the left side (e.g., 5 + Value).

        Args:
            other (float or int): The left-hand raw number.

        Returns:
            Value: A new node containing the sum.
        """
        return self + other

    def __rmul__(self, other):
        """
        Handles fallback multiplication when a raw number is on the left side (e.g., 5 * Value).

        Args:
            other (float or int): The left-hand raw number.

        Returns:
            Value: A new node containing the product.
        """
        return self * other

    def __rsub__(self, other):
        """
        Handles fallback subtraction when a raw number is on the left side (e.g., 5 - Value).

        Args:
            other (float or int): The left-hand raw number.

        Returns:
            Value: A new node containing the difference.
        """
        return other + (-self)

    def __pow__(self, n):
        """
        Raises this node to a constant integer or float power (self ** n).

        This executes the forward power step and defines the backward closure
        to route gradients using the power rule of calculus (n * x^(n-1)).

        Args:
            n (float or int): The exponent value. Note that this must be a fixed constant number,
                not another Value object.

        Returns:
            Value: A new node containing the powered result, linked back to its base parent node.
        """
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            """Applies the calculus power rule combined with the chain rule."""
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """
        Performs division by rewriting the operation as multiplying by a negative power (self * other**-1).

        Args:
            other (Value or float or int): The right-hand value to divide by.

        Returns:
            Value: A new node containing the quotient.
        """
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        """
        Applies the base-e exponential function to this node (e ** self).

        The derivative of e^x is famously just e^x itself. In the backward step, the incoming
        gradient is scaled directly by the forward result of this exponential operation.

        Returns:
            Value: A new node containing the exponential result, linked back to its parent node.
        """
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            """Applies the exponential derivative rule scaled by the chain rule."""
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        """
        Applies the natural logarithm (base e) function to this node.

        The derivative of log(x) is 1/x. In the backward step, the incoming gradient is divided
        by the original parent input data value before being accumulated.

        Returns:
            Value: A new node containing the logarithm result, linked back to its parent node.
        """
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            """Applies the logarithmic derivative rule (1/x) scaled by the chain rule."""
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        Applies the hyperbolic tangent (tanh) activation function to this node.

        This compresses any numerical input down into a smooth curve between -1.0 and 1.0.
        The derivative of tanh(x) is (1 - tanh(x)^2). The backward step uses this output curve
        value to scale the incoming gradient.

        Returns:
            Value: A new node containing the tanh output, linked back to its parent node.
        """
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            """Applies the tanh derivative curve formula scaled by the chain rule."""
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Executes a complete backpropagation pass starting from this node.

        This function automatically builds a strict, ordered list of all connected nodes
        using a topological sort (ensuring parent nodes appear after their children). It then
        sets this starting node's gradient to 1.0 and walks backward through the ordered list,
        triggering every single individual node's private _backward function to calculate
        all final derivatives across the entire computational graph.

        Returns:
            None: Modifies the '.grad' attributes of all connected nodes in place.
        """
        topo = []
        visited = set()

        def build_topo(v):
            """Recursively visits parent nodes to compile a safely ordered graph list."""
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
