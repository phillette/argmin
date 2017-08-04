import unittest
from pre_processing import prep_essays


class PrepEssaysTests(unittest.TestCase):
    def test_text_node_init(self):
        line = 'T13	Premise 1236 1301	they need to make some contribution ' \
               'too for their children future'
        node = prep_essays.TextNode(1, line)
        self.assertEqual(node.essay_number, 1)
        self.assertEqual(node.line, line)
        self.assertEqual(node.id, 12)
        self.assertEqual(node.type, 'Premise')
        self.assertEqual(node.text, 'they need to make some contribution too '
                                    'for their children future')

    def test_relations_init(self):
        pass

    def test_nodes_and_relations(self):
        pass

    def test_build_corpus_a(self):
        pass
