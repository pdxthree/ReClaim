class Trie:
    def __init__(self):
        self.root = {}
        self.sentence_end = -1
        
    def insert(self, sentence):
        curNode = self.root
        for token in sentence:
            if not token in curNode:
                curNode[token] = {}
            curNode = curNode[token]
        curNode[self.sentence_end] = True
 
    def search(self, sentence):
        curNode = self.root
        for token in sentence:
            if not token in curNode:
                return False
            curNode = curNode[token]
            
        if self.sentence_end not in curNode:
            return False
        return True
    
    def startsWith(self, prefix):
        curNode = self.root
        for c in prefix:
            if not c in curNode:
                return False
            curNode = curNode[c]
        
        return True
    
    def delete(self, sentence):
        nodes = [self.root]
        for token in sentence:
            if token in nodes[-1]:
                nodes.append(nodes[-1][token])
            else:
                return
        if self.sentence_end not in nodes[-1]:
            return
        
        nodes[-1].pop(self.sentence_end)
        for i in range(len(sentence)-1, -1, -1):
            if not nodes[i+1]:
                nodes[i].pop(sentence[i])
            else:
                break