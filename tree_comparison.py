# Below, two exemplary trees are shown,
# ... before explaining how the node-by-node comparison between two general decision trees of this syntax is performed:



# Exemplary, prototypical tree
# The decision trees to be analyzed for the comparison are available in this format
proto_example= {'Corr'     : 0,
        'Relationship_Var' : "Rel_Var_Muster_0",
        'DP'               : 100,
        'Node'             : {'Split': True, 'Split_Var': 'Split_Var_Muster_1', 'Split_Value' : "A"},

        'left':
            {'Corr'            : 0.1,
            'Relationship_Var' : "Rel_Var_Muster_1",
            'DP'               : 50,
            'Node'             : {'Split': True, 'Split_Var': "Split_Var_Muster_2", 'Split_Value' : 0.33},

             'left':
                {'Corr'            : 0.1,
                'Relationship_Var' : "Rel_Var_Muster_1",
                'DP'               : 50,
                'Node'             : {'Split': False}},

             'right':
                 {'Corr'            : 0.5,
                  'Relationship_Var': "Rel_Var_Muster_1",
                  'DP'              : 50,
                  'Node'            : {'Split': False}}
             },

                'right'            :
            {'Corr'            : 0.1,
            'Relationship_Var' : "Rel_Var_Muster_1",
            'DP'               : 50,
            'Node'             : {'Split': False}
             }
                }

# Exemplary, manifested tree (slight differences to the prototypical tree)
# The decision trees to be analyzed for the comparison are available in this format
manif_example   = {'Corr'       : 0,
         'Relationship_Var'     : "Rel_Var_Test_0",
         'DP'                   : 100,
         'Node'                 : {'Split': True, 'Split_Var': 'Split_Var_Muster_1', 'Split_Value': "B"},

          'left':
              {'Corr': 0.1,
               'Relationship_Var': "Rel_Var_Muster_1",
               'DP': 50,
               'Node': {'Split': True, 'Split_Var': "Split_Var_Muster_2", 'Split_Value': 0.1},

               'left':
                   {'Corr': 0.1,
                    'Relationship_Var': "Rel_Var_Muster_1",
                    'DP': 50,
                    'Node': {'Split': False}},

               'right':
                   {'Corr': 0.5,
                    'Relationship_Var': "Rel_Var_Muster_1",
                    'DP': 50,
                    'Node': {'Split': False}}
               },

                   'right':
             {'Corr'            : 1,
              'Relationship_Var': "Rel_Var_Test_1",
              'DP'              : 50,
              'Node'            : {'Split': True, 'Split_Var': "Split_Var_Muster_2", 'Split_Value': 0.2},
              'left':
                   {'Corr': 0.1,
                    'Relationship_Var': "Rel_Var_Muster_1",
                    'DP': 50,
                    'Node': {'Split': False}},
               'right':
                   {'Corr': 0.5,
                    'Relationship_Var': "Rel_Var_Muster_1",
                    'DP': 50,
                    'Node': {'Split': False}}
              }
                   }


#_______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________

# Function for comparing two nodes.
# The function ends with its recursive call to examine the nodes of the trees in the next level.
def compare_nodes(true_node, test_node, sampi_score_tracker, type="rel"):

    if type == "rel" and not true_node['Node']['Split']:
        return  # If “rel”, ignore nodes that only exist in the test tree.
    elif not true_node['Node']['Split'] and not test_node['Node']['Split']:
        return # If “abs”, ignore all nodes that do not lead to a split in either of the two trees

    # Another node is checked in the node-by-node comparison...
    sampi_score_tracker['n_nodes'] += 1

    # Check whether both nodes are splitting:
    if not (true_node['Node']['Split'] and test_node['Node']['Split']): # If only one of the two trees splits.
        sampi_score_tracker['sampi_score'] += 1
    else:# If there is a split in both trees at this node, the split variables are compared first:
        if true_node['Node']['Split_Var'] != test_node['Node']['Split_Var']:
            sampi_score_tracker['sampi_score'] += 1
        else:
            # Comparison of the split values. Three cases are distinguished:
            # - different class of the same categorical split variable
            # - the same continuous split variable
            # - one split variable is continuous, the other is categorical
            split_value_1 = true_node['Node'].get('Split_Value', 0)
            split_value_2 = test_node['Node'].get('Split_Value', 0)
            if isinstance(split_value_1, str) and split_value_1 != split_value_2:
                sampi_score_tracker['sampi_score'] += 0.5
            elif isinstance(split_value_1, (int, float)) and isinstance(split_value_2, (int, float)):
                sampi_score_tracker['sampi_score'] += min(abs(split_value_1 - split_value_2) * 2, 1)
            elif isinstance(split_value_1, (int, float)) and isinstance(split_value_2, str):
                sampi_score_tracker['sampi_score'] += 1


    # In order for the recursion to take place, missing nodes that only occur in one of the two trees ...
    # ...(depending on whether Sampi_rel or Sampi_abs is calculated) are replaced for the other tree ...
    # ... by the information that there is no branching at this node -> {'Node': {'Split': False}}

    # Recursion in the left subtrees;
    if 'left' in true_node:
        if 'left' in test_node:
            compare_nodes(true_node['left'], test_node['left'], sampi_score_tracker, type=type)
        else:    # If 'left' only exists in true_node:
            compare_nodes(true_node['left'], {'Node': {'Split': False}}, sampi_score_tracker, type=type)
    elif ('left' not in true_node) and ('left' in test_node) and (type == 'abs'):
        compare_nodes({'Node': {'Split': False}}, test_node['left'], sampi_score_tracker, type=type)

    # Recursion in the right subtrees:
    if 'right' in true_node:
        if 'right' in test_node:
            compare_nodes(true_node['right'], test_node['right'], sampi_score_tracker, type=type)
        else:   # If 'right' only exists in true_node:
            compare_nodes(true_node['right'], {'Node': {'Split': False}}, sampi_score_tracker, type=type)
    elif ('right' not in true_node) and ('right' in test_node) and (type == 'abs'):
        compare_nodes({'Node': {'Split': False}}, test_node['right'], sampi_score_tracker, type=type)



# Function for calculating the sampi value (Similarity of A Manifested and Prototype-tree Indicator)
# Depending on whether the relative or absolute sampi value is calculated,...
# ... the nodes to be examined are determined:
# (those that are present in at least the “true_tree” vs. those that are present in at least one of the two trees)
def calc_sampi(true_tree, test_tree, type="rel"):
    sampi_score_tracker = {'sampi_score': 0, 'n_nodes': 0}

    # Start node-by-node comparison of both trees at the root node.
    compare_nodes(true_tree, test_tree, sampi_score_tracker, type=type)

    # Normalize Sampi_Score with regard to the total number of examined nodes (calculate Sampi_rel resp. Sampi_abs).
    sampi_score = sampi_score_tracker['sampi_score']
    n_nodes = sampi_score_tracker['n_nodes']
    sampi = sampi_score / n_nodes if n_nodes > 0 else 0
    return sampi


# Example call for the function above:
# Sampi_rel and Sampi_abs are calculated for the exemplary trees (proto_example and manif_example, see above)
# print("\u03e1_rel = ", calc_sampi(proto_example, manif_example, type = "rel"))
# print("\u03e1_abs = ", calc_sampi(proto_example, manif_example, type = "abs"))


#_______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________
# A Directory of all Prototype Trees

# Template for prototypical trees with one splits
one_split = {
                 'Corr': None, 'Relationship_Var' : 'X', 'DP': 10000,
                 'Node': {'Split': True, 'Split_Var': None, "Split_Value": None},
                 'left':
                     {
                         'Corr': None, 'Relationship_Var' : 'X', 'DP': 1000,
                         'Node': {'Split': False}
                     },
                'right':
                    {
                        'Corr': None, 'Relationship_Var' : None, 'DP': 9000,
                        'Node': {'Split': False}
                    }
              }

# Template for prototypical trees with two splits:
two_split = {
                'Corr': None, 'Relationship_Var': 'X', 'DP' : 20000,
                'Node' : {'Split': True, 'Split_Var': None, 'Split_Value': None},
                'left':
                    {
                        'Corr': None, 'Relationship_Var': "X", 'DP': 10000,
                        'Node': {'Split': True, 'Split_Var': None, 'Split_Value': None},
                        'left':
                             {
                                'Corr': None, 'Relationship_Var': "X", 'DP': 1000,
                                'Node': {'Split': False}
                             },
                        'right':
                             {
                                 'Corr': None, 'Relationship_Var': None, 'DP': 9000,
                                 'Node': {'Split': False}
                             }
                     },
                'right':
                    {
                        'Corr': None, 'Relationship_Var': None, 'DP': 10000,
                        'Node': {'Split': False}
                    }
            }

# Template for prototypical trees with three splits:
three_split = {
                'Corr': None, 'Relationship_Var': 'X', 'DP' : 40000,
                'Node' : {'Split': True, 'Split_Var': None, 'Split_Value': None},
                'left':
                    {
                        'Corr': None, 'Relationship_Var': "X", 'DP': 20000,
                        'Node': {'Split': True, 'Split_Var': None, 'Split_Value': None},
                        'left':
                             {
                                'Corr': None, 'Relationship_Var': "X", 'DP': 10000,
                                'Node': {'Split': True, 'Split_Var': None, 'Split_Value': None},
                                'left':
                                    {
                                        'Corr': None, 'Relationship_Var': "X", 'DP': 1000,
                                        'Node': {'Split': False}
                                    },
                                'right':
                                    {
                                        'Corr': None, 'Relationship_Var': None, 'DP': 9000,
                                        'Node': {'Split': False}
                                    },
                             },
                        'right':
                             {
                                 'Corr': None, 'Relationship_Var': None, 'DP': 10000,
                                 'Node': {'Split': False}
                             }
                     },
                'right':
                    {
                        'Corr': None, 'Relationship_Var': None, 'DP': 20000,
                        'Node': {'Split': False}
                    }
            }


# Function supplements the templates of the prototypical trees with concrete values,...
# ... depending on the number of divisions and the corresponding division variables.
# Function passes the prototypical tree as a directory in the output (see above).
def prototype_trees(var1, var2, var3):
    if var1 == "O":  # Two or one splits
        if var2 == "O": # Tree with a single split
            tree = one_split
            tree['Node']['Split_Var'] = var3
            if var3 == "Kt":  # Categorical Split
                tree["Node"]['Split_Value'] = "A"
            else:             # Continuous Split (if var3 = X, Y or Kn)
                tree["Node"]['Split_Value'] = 0.1

        else: #Tree with two splits
            tree = two_split
            tree['Node']['Split_Var'] = var2
            if var2 == "Kt":  # Categorical Split
                tree["Node"]['Split_Value'] = "B"
            else:             # Continuous Split (if var2 = X, Y or Kn)
                tree["Node"]['Split_Value'] = 0.5

            tree['left']['Node']['Split_Var'] = var3
            if var3 == "Kt":  # Categorical Split
                tree['left']['Node']['Split_Value'] = "A"
            else:             # Continuous Split (if var3 = X, Y or Kn)
                tree['left']['Node']['Split_Value'] = 0.1

    else: #Tree with three splits
        tree = three_split
        tree['Node']['Split_Var'] = var1
        if var1 == "Kt":  # Categorical Split
            tree["Node"]['Split_Value'] = "C"
        else:  # Continuous Split (if var2 = X, Y or Kn)
            tree["Node"]['Split_Value'] = 0.5

        tree['left']['Node']['Split_Var'] = var2
        if var2 == "Kt":  # Categorical Split
            tree['left']['Node']['Split_Value'] = "B"
        else:  # Continuous Split (if var3 = X, Y or Kn)
            tree['left']['Node']['Split_Value'] = 0.5

        tree['left']['left']['Node']['Split_Var'] = var3
        if var3 == "Kt":  # Categorical Split
            tree['left']['left']['Node']['Split_Value'] = "A"
        else:  # Continuous Split (if var3 = X, Y or Kn)
            tree['left']['left']['Node']['Split_Value'] = 0.1

    return tree


# Test call for the creation of a prototypical tree:
# test = prototype_trees("Y", "Kt", "X")