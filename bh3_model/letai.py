from pysb import *
from pysb.macros import *
from pysb.macros import _macro_rule, _monomer_pattern_label

Model()

Monomer('Mito', ['dPsiM', 'jc1'],
        {'dPsiM': ['y', 'n'], 'jc1': ['y','n']})

Monomer('FCCP', [])

Monomer('Bim', ['state'], {'state': ['I', 'A']})
Monomer('Bid', ['state'], {'state': ['I', 'A']})
Monomer('M', ['state'], {'state': ['I','A']} )
Monomer('Pore', [])

# Initial Conditions (units in nanomolar)
Initial(Mito(dPsiM='y', jc1='n'), Parameter('Mito_0', 1))
Initial(Bim(state='I'), Parameter('Bim_0', 0))
Initial(Bid(state='I'), Parameter('Bid_0', 0))
Initial(FCCP(), Parameter('FCCP_0', 1000))
Initial(M(state='I'), Parameter('M_0', 10000))

# Dye uptake and spontaneous release:
Rule('dye_uptake',
      Mito(dPsiM='y', jc1='n') >> Mito(dPsiM='y', jc1='y'),
      #Parameter('dye_uptake_kf', 10**-2.37516935977979))
      Parameter('dye_uptake_kf', 0.03))
Rule('spontaneous_release',
     Mito(dPsiM='y', jc1=WILD) >> Mito(dPsiM='n', jc1='n'),
     Parameter('spontaneous_release_kf', 1e-3))
     #Parameter('spontaneous_release_kf', 10**-2.1060174548104249))

# We add a parameter for the time offset after treatment but before
# the start of measurement
#Parameter('t_offset', 10**0.92700389506712311)
Parameter('t_offset', 10)

# Observable: jc1 signal
Observable('jc1', Mito(jc1='y'))

# Macros for activation
# =====================
def simple_catalysis():
    """Follows the same scheme as the "simple catalysis" model in the
    Kuwana/Newmeyer paper, in which a catalyst acts directly to permeabilize
    the mitos.
    """
    catalyze_one_step(Bim(), Mito(), Mito(dPsiM='n', jc1='n', b=None),
            Parameter('Bim_kf', 2.7e-6))
    catalyze_one_step(Bid(), Mito(), Mito(dPsiM='n', jc1='n', b=None),
             Parameter('Bid_kf', 7e-6))

    # FCCP acts by simple "catalysis"
    catalyze_one_step(FCCP(), Mito(), Mito(dPsiM='n', jc1='n', b=None),
             Parameter('FCCP_kf', 1e-3))

def catalyst_activation():
    # First, have spontaneous "activation" of the catalyst
    Rule('Bim_activation', Bim(state='I') >> Bim(state='A'),
         Parameter('Bim_activation_kf', 1e-2))
    Rule('Bid_activation', Bid(state='I') >> Bid(state='A'),
         Parameter('Bid_activation_kf', 1e-2))

    # Now, have the activated catalyst permeabilize the mitos
    catalyze_one_step(Bim(state='A'), Mito(), Mito(dPsiM='n', jc1='n', b=None),
             Parameter('BimA_kf', 1e-5))
    catalyze_one_step(Bid(state='A'), Mito(), Mito(dPsiM='n', jc1='n', b=None),
             Parameter('BidA_kf', 1e-5))

    # FCCP acts by simple "catalysis"
    catalyze_one_step(FCCP(), Mito(), Mito(dPsiM='n', jc1='n', b=None),
             Parameter('FCCP_kf', 1e-3))

def catalyst_assembly():
    # The "monomer" is activated
    Rule('M_activation', M(state='I') >> M(state='A'),
         Parameter('M_activation_kf', 1e-1))

    # The pore assembles reversibly
    assemble_unstructured_pore(M(state='A'), Pore(), 8,
         [Parameter('pore_kf', 1e-3), Parameter('pore_kr', 1e-1)])
    
    # The pore catalyzes jc1 release
    catalyze_one_step(Pore(), Mito(), Mito(dPsiM='n', jc1='n'),
             Parameter('pore_mito_kf', 1e-2))

   # FCCP acts by simple "catalysis"
    catalyze_one_step(FCCP(), Mito(), Mito(dPsiM='n', jc1='n'),
             Parameter('FCCP_kf', 1))

def assemble_unstructured_pore(subunit, pore, order, klist):
    """Generate the order-n assembly reaction n*Subunit <> Pore."""

    # This is a function that is passed to macros._macro_rule to generate
    # the name for the pore assembly rule. It follows the pattern of,
    # e.g., "BaxA4_to_Pore" for a Bax pore of size 4.
    def pore_rule_name(rule_expression):
        react_p = rule_expression.reactant_pattern
        mp = react_p.complex_patterns[0].monomer_patterns[0]
        subunit_name = _monomer_pattern_label(mp)
        pore_name = mp.monomer.name 
        return '%s%d_to_%s' % (subunit_name, order, mp.monomer.name)
    
    # Assemble rule lhs
    lhs_pattern = subunit
    for i in range(1, order):
        lhs_pattern = lhs_pattern + subunit

    # Create the pore formation rule
    _macro_rule('unstructured_pore',
        lhs_pattern <> pore,
        klist, ['kf', 'kr'], name_func=pore_rule_name)

catalyst_assembly()


"""
def assemble_pore_spontaneous(subunit, order, klist):
    # Generate the order-n assembly reaction n*Subunit <> Pore.

    # This is a function that is passed to macros._macro_rule to generate
    # the name for the pore assembly rule. It follows the pattern of,
    # e.g., "BaxA_to_BaxA4" for a Bax pore of size 4.
    def pore_rule_name(rule_expression):
        react_p = rule_expression.reactant_pattern
        mp = react_p.complex_patterns[0].monomer_patterns[0]
        subunit_name = macros._monomer_pattern_label(mp)
        pore_name = mp.monomer.name 
        return '%s_to_%s%d' % (subunit_name, mp.monomer.name, 4)
    
    # Alias for a subunit that is capable of forming a pore
    free_subunit = subunit(s1=None, s2=None)

    # Assemble rule lhs
    lhs_pattern = free_subunit
    for i in range(1, order):
        lhs_pattern = pattern + free_subunit

    # Assemble rule rhs
    rhs_pattern = subunit(s1=1, s2=order)
    l_bond_index = 2 
    r_bond_index = 1
    while l_bond_index <= order:
        rhs_pattern = rhs_pattern % subunit(s1=l_bond_index, s2=r_bond_index)
        l_bond_index += 1
        r_bond_index += 1

    # Create the pore formation rule
    macros._macro_rule('spontaneous_pore',
        lhs_pattern <> rhs_pattern,
        klist, ['kf', 'kr'], name_func=pore_rule_name)
"""

