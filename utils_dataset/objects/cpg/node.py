from .properties import Properties
from .edge import Edge
from ... import log as logger

node_labels = ["Block", "Call", "Comment", "ControlStructure", "File", "Identifier", "FieldIdentifier", "Literal",
               "Local", "Member", "MetaData", "Method", "MethodInst", "MethodParameterIn", "MethodParameterOut",
               "MethodReturn", "Namespace", "NamespaceBlock", "Return", "Type", "TypeDecl", "Unknown"]

operators = ['addition', 'addressOf', 'and', 'arithmeticShiftRight', 'assignment',
             'assignmentAnd', 'assignmentArithmeticShiftRight', 'assignmentDivision',
             'assignmentMinus', 'assignmentMultiplication', 'assignmentOr', 'assignmentPlus',
             'assignmentShiftLeft', 'assignmentXor', 'cast', 'conditionalExpression',
             'division', 'equals', 'fieldAccess', 'greaterEqualsThan', 'greaterThan',
             'indirectFieldAccess', 'indirectIndexAccess', 'indirection', 'lessEqualsThan',
             'lessThan', 'logicalAnd', 'logicalNot', 'logicalOr', 'minus', 'modulo', 'multiplication',
             'not', 'notEquals', 'or', 'postDecrement', 'plus', 'postIncrement', 'preDecrement',
             'preIncrement', 'shiftLeft', 'sizeOf', 'subtraction']

node_labels += operators

node_labels = {label: i for i, label in enumerate(node_labels)}

PRINT_PROPS = True


class Node:
    def __init__(self, node, indentation):
        self.id = node["id"].split(".")[-1]
        self.node_type = self.id.split("[")[0]
        self.label = self.id.split("[")[0]
        self.indentation = indentation + 1
        self.properties = Properties(node["properties"], self.indentation)
        self.edges = {edge["id"].split(".")[-1]: Edge(edge, self.indentation) for edge in node["edges"]}
        self.order = None
        operator = self.properties.get_operator()
        self.label = operator if operator is not None else self.label
        self._set_type()

    def __str__(self):
        indentation = self.indentation * "\t"
        properties = f"{indentation}Properties: {self.properties}\n"
        edges_str = ""

        for edge in self.edges:
            edges_str += f"{self.edges[edge]}"

        return f"\n{indentation}Node id: {self.id}\n{properties if PRINT_PROPS else ''}{indentation}Edges: {edges_str}"

    def connections(self, connections, e_type):
        for e_id, edge in self.edges.items():
            if edge.type != e_type: continue

            if edge.node_in in connections["in"] and edge.node_in != self.id:
                connections["in"][self.id] = edge.node_in

            if edge.node_out in connections["out"] and edge.node_out != self.id:
                connections["out"][self.id] = edge.node_out

        return connections

    def has_code(self):
        return self.properties.has_code()

    def has_line_number(self):
        return self.properties.has_line_number()

    def get_code(self):
        return self.properties.code()

    def get_line_number(self):
        return self.properties.line_number()

    def get_column_number(self):
        return self.properties.column_number()

    def _set_type(self):
        # label = self.label if self.operator is None else self.operator
        self.type = node_labels.get(self.label)  # Label embedding

        if self.type is None:
            logger.log_warning("node", f"LABEL {self.label} not in labels!")
            self.type = len(node_labels) + 1
    

    def get_ast_parents(self):
        parent_list = []
        ast_edge_list = []        
        edge_list = self.edges
        for edge in edge_list.keys():
            if edge.split('@')[0] == 'Ast':
                ast_edge_list.append(edge_list[edge])
        for ast_edge in ast_edge_list:
            node_in_id = ast_edge.node_in
            node_out_id = ast_edge.node_out
            if node_in_id == self.id:           
                parent_list.append(node_out_id)
        return parent_list[0]   # 默认一个节点只有一个ast的父节点     

    def has_ddg_edge(self):
        ddg_list = []
        edge_dict = self.edges
        for edge in edge_dict:
            if edge.split('@')[0] == 'Ddg':
                ddg_list.append(edge)
        if len(ddg_list) == 0:
            return None
        elif len(ddg_list) == 1:
            if ddg_list[0].split("[")[0] == 'MethodReturn':
                return None
        return ddg_list   

    def ddg_successors(self):
        successors_list = []
        ddg_list = []

        if self.label != 'Call':
            edge_dict = self.edges
            for edge in edge_dict:
                if edge.split('@')[0] == 'Ddg':
                    ddg_list.append(edge)

        else:
            edge_dict = self.edges
            for edge in edge_dict:
                if edge.split('@')[0] == 'Ddg':
                    ddg_list.append(edge)

        for ddg_edge in ddg_list:
            node_in_id = edge_dict[ddg_edge].node_in
            node_out_id = edge_dict[ddg_edge].node_out
            if node_in_id == self.id:
                if node_out_id not in successors_list:
                    successors_list.append(node_out_id)
        return successors_list
    
    def ddg_predecessors(self):
        predecessors_list = []
        ddg_list = []

        edge_dict = self.edges
        for edge in edge_dict:
            if edge.split('@')[0] == 'Ddg':
                ddg_list.append(edge)   
                         
        for ddg_edge in ddg_list:
            node_in_id = edge_dict[ddg_edge].node_in
            node_out_id = edge_dict[ddg_edge].node_out
            if node_out_id == self.id:
                if node_in_id not in predecessors_list:
                    predecessors_list.append(node_in_id)
        return predecessors_list

    def cdg_successors(self):
        successors_list = []
        cdg_list = []
        edge_dict = self.edges
        for edge in edge_dict:
            if edge.split('@')[0] == 'Cdg':
                cdg_list.append(edge)
        for cdg_edge in cdg_list:
            node_in_id = edge_dict[cdg_edge].node_in
            node_out_id = edge_dict[cdg_edge].node_out
            if node_in_id == self.id:
                if node_out_id not in successors_list:
                    successors_list.append(node_out_id)
        return successors_list
    
    def cdg_predecessors(self):
        predecessors_list = []
        cdg_list = []
        edge_dict = self.edges
        for edge in edge_dict:
            if edge.split('@')[0] == 'Cdg':
                edge_dict.append(edge)
        for cdg_edge in cdg_list:
            node_in_id = edge_dict[cdg_edge].node_in
            node_out_id = edge_dict[cdg_edge].node_out
            if node_out_id == self.id:
                if node_in_id not in predecessors_list:
                    predecessors_list.append(node_in_id)
        return predecessors_list