import numpy as np

codes = dict()

class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.code = ''

def huffman_encoding(z):
    nodes = create_tree(z)
    huffman = encoding(nodes[0])
    encoded_output = output_encoded(z,huffman)
    # print("symbols with codes", huffman)
    return encoded_output, nodes[0]

def create_tree(z):
    dict_data = frequency(z)
    symbols = dict_data.keys()
    freq = dict_data.values()

    nodes = []
    
    # converting symbols and freq into huffman tree nodes
    for symbol in symbols:
        node = Node(dict_data.get(symbol), symbol)
        nodes.append(node)

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)
    
        # pick 2 smallest nodes
        left = nodes[0]
        right = nodes[1]
    
        left.code = 0
        right.code = 1
    
        # combine the 2 smallest nodes to create new node
        root = Node(left.freq+right.freq, left.symbol+right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(root)

    return nodes

def encoding(node, val=''):
    # huffman code for current node
    newVal = val + str(node.code)

    if node.left:
        encoding(node.left, newVal)
    if node.right:
        encoding(node.right, newVal)

    if not node.left and not node.right:
        codes[node.symbol] = newVal
         
    return codes   

def frequency(z):
    # code = dict()
    # for element in z:
    #     if code.get(element) == None:
    #         code[element] = 1
    #     else:
    #         code[element] += 1
    
    unique, counts = np.unique(z, return_counts=True)
    code = dict(zip(unique, counts))
    return code

def output_encoded(data, coding):
    encoding_output = []
    for c in data:
      #  print(coding[c], end = '')
        encoding_output.append(coding[c])

    string = ''.join([str(item) for item in encoding_output])
    return string

# Huffman Decoding
def huffman_decoding(encoded_data, huffman_tree):
    tree_head = huffman_tree
    decoded_output = []
    for x in encoded_data:
        if x == '1':
            huffman_tree = huffman_tree.right   
        elif x == '0':
            huffman_tree = huffman_tree.left

        # if huffman_tree.left == None and huffman_tree.right == None:
        #     decoded_output.append(huffman_tree.symbol)
        #     # print("decoded_output: ", decoded_output)
        #     # print(huffman_tree.symbol)
        #     huffman_tree = tree_head

        try:
            if huffman_tree.left.symbol == None and huffman_tree.right.symbol == None:
                pass
        except AttributeError:
            decoded_output.append(huffman_tree.symbol)
            huffman_tree = tree_head

    return decoded_output

# """ First Test """
# data = np.array(['10', '10', '10', '10', '10', '9', '9', '9', '8','7','7','6','6','6','6'])

# print(data)
# encoding, tree = huffman_encoding(data)
# print("Encoded output", encoding)
# print("Decoded Output", huffman_decoding(encoding,tree))