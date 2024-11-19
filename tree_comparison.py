# Das Format sollen meine Trees haben (Muster):
Muster= {'Corr'            : 0,
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

# Das Format sollen meine Trees haben (Test-Baum mit leicht veränderten Werten bzgl. des Splits):
Test   = {'Corr'                : 0,
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
              'Node'            : {'Split': False}
              }
         }


#_______________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________


# Funktion zum Vergleich von zwei Knoten
def compare_nodes(true_node, test_node, sampi_score_tracker):
    # Prüfung, ob beide Knoten Splits enthalten
    if true_node['Node']['Split']:
        sampi_score_tracker['n_nodes'] += 1

        # Vergleich, ob beide Bäume splitten.
        if not test_node['Node']['Split']:
            sampi_score_tracker['sampi_score'] += 1      # Strafpunkt, da im Muster-Baum ein Split vorliegt, im Test-Baum aber nicht.
        else:
            # Vergleich der Split-Variablen
            if true_node['Node'].get('Split_Var') != test_node['Node'].get('Split_Var'):
                sampi_score_tracker['sampi_score']     += 1  # Strafpunkt für unterschiedliche Split-Variablen
            else:
                # Vergleich der Split-Werte
                split_value_1 = true_node['Node'].get('Split_Value', 0)
                split_value_2 = test_node['Node'].get('Split_Value', 0)
                if (type(split_value_1) == str) and (split_value_1 != split_value_2):       # Muster_Node ist "str", aber Test_Node ist nicht identisch
                    sampi_score_tracker['sampi_score']      += 1
                elif (type(split_value_1) != str) and (type(split_value_2) != str):         # Muster_Node und Test_Node sind Zahlen
                    sampi_score_tracker['sampi_score']      += abs(split_value_1 - split_value_2)   # Strafpunkte für Unterschied in Split-Werten
                elif (type(split_value_1) != str) and (type(split_value_2) == str):         # Muster_Node ist Zahl, aber Test_Node nicht.
                    sampi_score_tracker['sampi_score']      += 1

    # Vergleich der linken Teilbäume, falls vorhanden
    if 'left' in true_node:
        if 'left' in test_node:
            compare_nodes(true_node['left'], test_node['left'], sampi_score_tracker)
        else: # If 'left' only exists in true_node
            compare_nodes(true_node['left'], {'Node': {'Split': False}}, sampi_score_tracker)

    # Vergleich der linken Teilbäume, falls vorhanden
    if 'right' in true_node:
        if 'right' in test_node:
            compare_nodes(true_node['right'], test_node['right'], sampi_score_tracker)
        else: # If 'left' only exists in true_node
            compare_nodes(true_node['right'], {'Node': {'Split': False}}, sampi_score_tracker)



# Funktion zur Berechnung des Sampi-Werts. (Similarity of A Manifested and Prototype-tree Indicator)
def calc_sampi(tree1, tree2):
    sampi_score_tracker = {'sampi_score': 0, 'n_nodes': 0}

    # Vergleich der Wurzelknoten starten
    compare_nodes(tree1, tree2, sampi_score_tracker)

    # Sampi_Score hinsichtlich der Gesamtzahl an untersuchten Knoten normieren (Sampi berechnen).
    sampi_score = sampi_score_tracker['sampi_score']
    n_nodes     = sampi_score_tracker['n_nodes']
    sampi = sampi_score / n_nodes if n_nodes > 0 else 0
    return sampi

print("\u03e1 = ", calc_sampi(Muster, Test))