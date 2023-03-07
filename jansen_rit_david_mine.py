# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
A contributed model: The Jansen and Rit model as presented in (David et al., 2003) [JansenRitDavid2003];
and an extension to build a thalamocortical network combining the hierarchical modeling approach in
David et al. (2005) and the differential thalamo-cortical and cortico-cortical connectivity role
as proposed in Jones (2009).

@01/08/2022 Adding Wendling (2002) model for epilepsy.

.. moduleauthor:: Jesús Cabrera-Álvarez (jescab01@ucm.es)

"""

import numpy

from tvb.simulator.common import get_logger
from tvb.basic.neotraits.api import NArray, Range, Final, List
import tvb.simulator.models as models

LOG = get_logger(__name__)


class JansenRitDavid2003_th(models.Model):
    """
    Extending the extension.

    Here, we propose to use the Jansen-Rit's model extension (David et al., 2003) to enhance spectral richness.
    And to combine David et al. (2005) hierarchical Jansen-Rit with the Thalamic relevance posed in Jones (2009).

    The hierarchical implementation allowed to give different weights to connections coming from different
    cortical layers. We will take advantage of that proposal but using the scheme to differentiate the
    inputs from cortico-cortical connections and the input from thalamo-cortical connections.

    David et al. didnt consider thalamo-cortical connections as: "they represent a minority of extrinsic connections:
    it is thought that at least 99% of axons in white matter link cortical areas of the same hemisphere."

    TODO:


    """

    # Define traited attributes for this model, these represent possible kwargs.
    He1 = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    Hi1 = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    tau_e1 = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms) in the first kinetic population.""")

    tau_i1 = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms) in the first kinetic population.""")

    He2 = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain in the second kinetic population.""")

    Hi2 = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain in the second kinetic population.""")

    tau_e2 = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms) in the second kinetic population""")

    tau_i2 = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms) in the second kinetic population.""")

    w = NArray(
        label=r":math:`w`",
        default=numpy.array([0.8]),
        domain=Range(lo=0., hi=1.0, step=0.05),
        doc="""Relative proportion of each kinectic population in the cortical area. 
        Multiplies population 1; (1-w)*population 2""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([6.0]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2 * 2.5 [s^-1]; we convert to [ms^-1]""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    c = NArray(
        label=r":math:`c`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    c_pyr2exc = NArray(
        label=r":math:`c_pyr2exc`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop. From pyramidal cells to 
        excitatory interneurons. It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_exc2pyr`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. From excitatory 
        interneurons to pyramidal cells. It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_pyr2inh`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop. From pyramidal cells to 
        inhibitory interneurons. It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_inh2pyr`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop. From inhibitory cells
        to pyramidal cells. It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")


    p = NArray(
        label=r":math:`\p_{mean}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")


    # k = NArray(
    #     label=r":math:`k1`",
    #     default=numpy.array([0.5]),
    #     domain=Range(lo=0.0, hi=1.0, step=0.05),
    #     doc="""Contribution of simulated areas on ROI signal. +
    #     In contrast, (1-k) contribution of unknown elements to signal (noise).""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        {
            "vExc1": numpy.array([-1.0, 1.0]),
            "xExc1": numpy.array([-2.0, 2.0]),
            "vInh1": numpy.array([-5.0, 5.0]),
            "xInh1": numpy.array([-5.0, 5.0]),
            "vPyr1": numpy.array([-1.0, 1.0]),
            "xPyr1": numpy.array([-6.0, 6.0]),

            "vExc2": numpy.array([-1.0, 1.0]),
            "xExc2": numpy.array([-2.0, 2.0]),
            "vInh2": numpy.array([-5.0, 5.0]),
            "xInh2": numpy.array([-5.0, 5.0]),
            "vPyr2": numpy.array([-1.0, 1.0]),
            "xPyr2": numpy.array([-6.0, 6.0]),
        },
        label="State Variable ranges [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vExc1", "xExc1", "vInh1", "xInh1", "vPyr1", "xPyr1",
                 "vExc2", "xExc2", "vInh2", "xInh2", "vPyr2", "xPyr2"),

        default=("vExc1", "vInh1", "vPyr1", "vExc2", "vInh2", "vPyr2"),

        doc="""This represents the default state-variables of this Model to be
            monitored. It can be overridden for each Monitor if desired. 
            Correspondance in David 2003:
            vExc, xExc = v1, x1; vInh, xInh = v2, x2; vPyr, xPyr = v3, x3.
            Correspondance in Jansen-Rit 1995:
            vExc, xExc = y1, y4; vInh, xInh = y2, y5; vPyr, xPyr = y0, y3.""")

    state_variables = ["vExc1", "xExc1", "vInh1", "xInh1", "vPyr1", "xPyr1",
                       "vExc2", "xExc2", "vInh2", "xInh2", "vPyr2", "xPyr2"]
    _nvar = 12
    cvar = numpy.array([0, 2, 6, 8], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vExc1 = state_variables[0, :]
        xExc1 = state_variables[1, :]
        vInh1 = state_variables[2, :]
        xInh1 = state_variables[3, :]
        vPyr1 = state_variables[4, :]
        xPyr1 = state_variables[5, :]

        vExc2 = state_variables[6, :]
        xExc2 = state_variables[7, :]
        vInh2 = state_variables[8, :]
        xInh2 = state_variables[9, :]
        vPyr2 = state_variables[10, :]
        xPyr2 = state_variables[11, :]



        # NOTE for local couplings:
            # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
            # vInh, xInh inhibitory interneurons; vPyr, xPyr pyramidal neurons
        sum_vPyr = self.w * vPyr1 + (1-self.w) * vPyr2
        sum_vExc_vInh = self.w * (vExc1 - vInh1) + (1-self.w) * (vExc2 - vInh2)


        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        ## Weighted types of long range connections: [af - FeedForward - thalamo-cortical - proximal];
        # [ab - FeedBack - cortico-cortical - distal] Weights already applied in Coupling function.
        ab_lrc_fromcx = coupling[0, :]  # just cortico-cortical
        af_lrc_fromth = coupling[1, :]  # just thalamus

        src = local_coupling * (sum_vExc_vInh)

        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        S_pyr = (self.e0) / (1 + numpy.exp(self.r * (self.v0 - sum_vExc_vInh)))
        S_exc = (self.c * self.c_exc2pyr * self.e0) / (1 + numpy.exp(self.r * (self.v0 - self.c * self.c_pyr2exc * sum_vPyr)))
        S_inh = (self.c * self.c_inh2pyr * self.e0) / (1 + numpy.exp(self.r * (self.v0 - self.c * self.c_pyr2inh * sum_vPyr)))


        ### SLOW kinetic population
        dvExc1 = xExc1
        dxExc1 = self.He1 / self.tau_e1 * (input + S_exc + src + af_lrc_fromth) - (2 * xExc1) / self.tau_e1 - (vExc1 / self.tau_e1**2)
        dvInh1 = xInh1
        dxInh1 = self.Hi1 / self.tau_i1 * (S_inh + ab_lrc_fromcx) - (2 * xInh1) / self.tau_i1 - (vInh1 / self.tau_i1**2)
        dvPyr1 = xPyr1
        dxPyr1 = self.He1 / self.tau_e1 * (S_pyr + ab_lrc_fromcx) - (2 * xPyr1) / self.tau_e1 - (vPyr1 / self.tau_e1**2)

        ### FAST kinetic population
        dvExc2 = xExc2
        dxExc2 = self.He2 / self.tau_e2 * (input + S_exc + src + af_lrc_fromth) - (2 * xExc2) / self.tau_e2 - (vExc2 / self.tau_e2**2)
        dvInh2 = xInh2
        dxInh2 = self.Hi2 / self.tau_i2 * (S_inh + ab_lrc_fromcx) - (2 * xInh2) / self.tau_i2 - (vInh2 / self.tau_i2**2)
        dvPyr2 = xPyr2
        dxPyr2 = self.He2 / self.tau_e2 * (S_pyr + ab_lrc_fromcx) - (2 * xPyr2) / self.tau_e2 - (vPyr2 / self.tau_e2**2)

        derivative = numpy.array([dvExc1, dxExc1, dvInh1, dxInh1, dvPyr1, dxPyr1,
                                  dvExc2, dxExc2, dvInh2, dxInh2, dvPyr2, dxPyr2])

        return derivative


class JansenRitDavid2003(models.Model):
    """
    The Jansen and Rit model as studied by David et al., 2003
    They showed how an extension of Jansen-Rit could enhance simulation spectrum to look closer to MEG recordings.

    Their extension consisted on introducing a gamma oscillator coupled to a main alpha oscillator into each subnode.
    Couping the activity of those two oscillators resulted into broader alpha spectra, closer to reality.

    TODO:
    - Interregional coupling

    """

    # Define traited attributes for this model, these represent possible kwargs.
    He1 = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    Hi1 = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain in the first kinetic population.""")

    tau_e1 = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms) in the first kinetic population.""")

    tau_i1 = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms) in the first kinetic population.""")

    He2 = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain in the second kinetic population.""")

    Hi2 = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain in the second kinetic population.""")

    tau_e2 = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms) in the second kinetic population""")

    tau_i2 = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms) in the second kinetic population.""")

    w = NArray(
        label=r":math:`w`",
        default=numpy.array([0.8]),
        domain=Range(lo=0., hi=1.0, step=0.05),
        doc="""Relative proportion of each kinectic population in the cortical area. 
        Multiplies population 1; (1-w)*population 2""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([6.0]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2 * 2.5 [s^-1]; we convert to [ms^-1]""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    c = NArray(
        label=r":math:`c`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    c_pyr2exc = NArray(
        label=r":math:`c_pyr2exc`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop. From pyramidal cells to 
        excitatory interneurons. It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_exc2pyr`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. From excitatory 
        interneurons to pyramidal cells. It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_pyr2inh`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop. From pyramidal cells to 
        inhibitory interneurons. It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_inh2pyr`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop. From inhibitory cells
        to pyramidal cells. It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")


    p = NArray(
        label=r":math:`\p_{mean}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.32, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",

        default={"vPyr1": numpy.array([-1.0, 1.0]),
                 "vExc1": numpy.array([-500.0, 500.0]),
                 "vInh1": numpy.array([-50.0, 50.0]),
                 "xPyr1": numpy.array([-6.0, 6.0]),
                 "xExc1": numpy.array([-20.0, 20.0]),
                 "xInh1": numpy.array([-500.0, 500.0]),

                 "vPyr2": numpy.array([-1.0, 1.0]),
                 "vExc2": numpy.array([-500.0, 500.0]),
                 "vInh2": numpy.array([-50.0, 50.0]),
                 "xPyr2": numpy.array([-6.0, 6.0]),
                 "xExc2": numpy.array([-20.0, 20.0]),
                 "xInh2": numpy.array([-500.0, 500.0])},

        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr1", "vExc1", "vInh1", "xPyr1", "xExc1", "xInh1",
                 "vPyr2", "vExc2", "vInh2", "xPyr2", "xExc2", "xInh2"),

        default=("vPyr1", "vExc1", "vInh1", "xPyr1",
                 "vPyr2", "vExc2", "vInh2", "xPyr2"),

        doc="""This represents the default state-variables of this Model to be
            monitored. It can be overridden for each Monitor if desired. 
            Correspondance in David 2003:
            vExc, xExc = v1, x1; vInh, xInh = v2, x2; vPyr, xPyr = v3, x3.
            Correspondance in Jansen-Rit 1995:
            vExc, xExc = y1, y4; vInh, xInh = y2, y5; vPyr, xPyr = y0, y3.""")

    state_variables = ["vPyr1", "vExc1", "vInh1", "xPyr1", "xExc1", "xInh1",
                       "vPyr2", "vExc2", "vInh2", "xPyr2", "xExc2", "xInh2"]
    _nvar = 12
    cvar = numpy.array([1, 2, 7, 8], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr1 = state_variables[0, :]
        vExc1 = state_variables[1, :]
        vInh1 = state_variables[2, :]
        xPyr1 = state_variables[3, :]
        xExc1 = state_variables[4, :]
        xInh1 = state_variables[5, :]

        vPyr2 = state_variables[6, :]
        vExc2 = state_variables[7, :]
        vInh2 = state_variables[8, :]
        xPyr2 = state_variables[9, :]
        xExc2 = state_variables[10, :]
        xInh2 = state_variables[11, :]

        sum_vPyr = self.w * vPyr1 + (1-self.w) * vPyr2
        sum_vExc_vInh = self.w * (vExc1 - vInh1) + (1-self.w) * (vExc2 - vInh2)

        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        lrc = coupling[0, :]
        src = local_coupling * (sum_vExc_vInh)

        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        S_pyr = self.e0 / (1 + numpy.exp(self.r * (self.v0 - sum_vExc_vInh)))
        S_exc = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2exc * self.c * sum_vPyr)))
        S_inh = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inh * self.c * sum_vPyr)))

        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons
        # vPyr, xPyr pyramidal neurons

        ### SLOW kinetic population
        dvPyr1 = xPyr1
        dvExc1 = xExc1
        dvInh1 = xInh1
        dxPyr1 = self.He1 / self.tau_e1 * S_pyr - (2 * xPyr1) / self.tau_e1 - vPyr1 / self.tau_e1 ** 2
        dxExc1 = self.He1 / self.tau_e1 * (input + self.c_exc2pyr * self.c * S_exc + lrc + src) - (2 * xExc1) / self.tau_e1 - vExc1 / self.tau_e1 ** 2
        dxInh1 = self.Hi1 / self.tau_i1 * (self.c_inh2pyr * self.c * S_inh) - (2 * xInh1) / self.tau_i1 - vInh1 / self.tau_i1 ** 2

        ### FAST kinetic population
        dvPyr2 = xPyr2
        dvExc2 = xExc2
        dvInh2 = xInh2
        dxPyr2 = self.He2 / self.tau_e2 * S_pyr - (2 * xPyr2) / self.tau_e2 - vPyr2 / self.tau_e2 ** 2
        dxExc2 = self.He2 / self.tau_e2 * (input + self.c_exc2pyr * self.c * S_exc + lrc + src) - (2 * xExc2) / self.tau_e2 - vExc2 / self.tau_e2 ** 2
        dxInh2 = self.Hi2 / self.tau_i2 * (self.c_inh2pyr * self.c * S_inh) - (2 * xInh2) / self.tau_i2 - vInh2 / self.tau_i2 ** 2

        derivative = numpy.array([dvPyr1, dvExc1, dvInh1, dxPyr1, dxExc1, dxInh1,
                                  dvPyr2, dvExc2, dvInh2, dxPyr2, dxExc2, dxInh2])

        return derivative


class JansenRit1995(models.Model):
    """
    This is a version of TVB Jansen-Rit model; in which I just change variable names
    to get an easier to read model; as baseline for further work on it:
        - adding specific noise to thalamus
        - building JansenRitDavid model above.
    """

    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms)""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 
        6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2.5 [s^-1]; we convert to [ms^-1]""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    c = NArray(
        label=r":math:`c`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    c_pyr2exc = NArray(
        label=r":math:`c_11`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_12`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. 
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_21`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"vPyr": numpy.array([-1.0, 1.0]),
                 "vExc": numpy.array([-500.0, 500.0]),
                 "vInh": numpy.array([-50.0, 50.0]),
                 "xPyr": numpy.array([-6.0, 6.0]),
                 "xExc": numpy.array([-20.0, 20.0]),
                 "xInh": numpy.array([-500.0, 500.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"),
        default=("vPyr", "vExc", "vInh", "xPyr"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh"]
    _nvar = 6
    cvar = numpy.array([1, 2], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr = state_variables[0, :]
        vExc = state_variables[1, :]
        vInh = state_variables[2, :]
        xPyr = state_variables[3, :]
        xExc = state_variables[4, :]
        xInh = state_variables[5, :]

        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        lrc = coupling[0, :]
        src = local_coupling * (vExc - vInh)

        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        S_pyr = self.e0 / (1 + numpy.exp(self.r * (self.v0 - (vExc - vInh))))
        S_exc = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2exc * self.c * vPyr)))
        S_inh = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inh * self.c * vPyr)))

        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons
        # vPyr, xPyr pyramidal neurons

        dvPyr = xPyr
        dvExc = xExc
        dvInh = xInh
        dxPyr = self.He / self.tau_e * S_pyr - (2 * xPyr) / self.tau_e - vPyr / self.tau_e ** 2
        dxExc = self.He / self.tau_e * (input + self.c_exc2pyr * self.c * S_exc + lrc + src) - (2 * xExc) / self.tau_e - vExc / self.tau_e ** 2
        dxInh = self.Hi / self.tau_i * (self.c_inh2pyr * self.c * S_inh) - (2 * xInh) / self.tau_i - vInh / self.tau_i ** 2

        derivative = numpy.array([dvPyr, dvExc, dvInh, dxPyr, dxExc, dxInh])

        return derivative


class Wendling2002(models.Model):
    """
    Wendling 2002 . Epileptic fast activity can be explained by a model of impaired GABAergic dendritic inhibition.

    """

    # Define traited attributes for this model, these represent possible kwargs.
    He = NArray(
        label=":math:`He`",
        default=numpy.array([3.25]),
        domain=Range(lo=2.6, hi=9.75, step=0.05),
        doc="""Maximum amplitude of EPSP [mV]. Also called average synaptic gain.""")

    Hi = NArray(
        label=":math:`B`",
        default=numpy.array([22.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    Hif = NArray(
        label=":math:`G`",
        default=numpy.array([10.0]),
        domain=Range(lo=17.6, hi=110.0, step=0.2),
        doc="""fast inhibitory Maximum amplitude of IPSP [mV]. Also called average synaptic gain.""")

    tau_e = NArray(
        label=":math:`a`",
        default=numpy.array([10.0]),
        domain=Range(lo=6.3, hi=20.0, step=0.1),
        doc="""Excitatory time constant (ms)""")

    tau_i = NArray(
        label=":math:`b`",
        default=numpy.array([20.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""Inhibitory time constant (ms)""")

    tau_if = NArray(
        label=":math:`g`",
        default=numpy.array([2.0]),
        domain=Range(lo=12.0, hi=40.0, step=0.2),
        doc="""fast Inhibitory time constant (ms)""")

    v0 = NArray(
        label=":math:`v_0`",
        default=numpy.array([5.52]),
        domain=Range(lo=3.12, hi=6.0, step=0.02),
        doc="""Firing threshold (PSP) for which a 50% firing rate is achieved.
        In other words, it is the value of the average membrane potential
        corresponding to the inflection point of the sigmoid [mV]; 
        6.0 in JansenRit1995 and DavidFriston2003""")

    e0 = NArray(
        label=":math:`e_0`",
        default=numpy.array([0.005]),
        domain=Range(lo=0.00125, hi=0.00375, step=0.00001),
        doc="""Determines the maximum firing rate of the neural population
        [ms^-1]. In the papers its value is 2.5 [s^-1]; we convert to [ms^-1]""")

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.56]),
        domain=Range(lo=0.28, hi=0.84, step=0.01),
        doc="""Steepness of the sigmoidal transformation [mV^-1].""")

    c = NArray(
        label=r":math:`c`",
        default=numpy.array([135.0]),
        domain=Range(lo=65.0, hi=1350.0, step=1.),
        doc="""Average number of synapses between populations.""")

    c_pyr2exc = NArray(
        label=r":math:`c_11`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""Average probability of synaptic contacts in the feedback excitatory loop.
        It multiplies c; so c_11 = c = 135 contacts""")

    c_exc2pyr = NArray(
        label=r":math:`c_12`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.4, hi=1.2, step=0.1),
        doc="""Average probability of synaptic contacts in the slow feedback excitatory loop. 
        It multiplies c; so c_12 = c * 0.8 = 108 contacts""")

    c_pyr2inh = NArray(
        label=r":math:`c_21`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.25]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts in the slow feedback inhibitory loop.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_pyr2inhf = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.3]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts from pyramidal to fast inhibitory.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inh2inhf = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts from slow inhibitory to fast inhibitory.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    c_inhf2pyr = NArray(
        label=r":math:`c_22`",
        default=numpy.array([0.8]),
        domain=Range(lo=0.125, hi=0.375, step=0.005),
        doc="""Average probability of synaptic contacts from fast inhibitory to pyramidals.
        It multiplies c; so c_21 = c_22 = c * 0.25 = 33.75 contacts""")

    p = NArray(
        label=r":math:`\p_{max}`",
        default=numpy.array([0.22]),
        domain=Range(lo=0.0, hi=0.22, step=0.01),
        doc="""Mean input firing rate""")

    sigma = NArray(
        label=r":math:`\sigma`",
        default=numpy.array([0.022]),
        domain=Range(lo=0.0, hi=0.05, step=0.005),
        doc="""Standard deviation of input firing rate following a Gaussian""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"vPyr": numpy.array([-1.0, 1.0]),
                 "vExc": numpy.array([-500.0, 500.0]),
                 "vInh": numpy.array([-50.0, 50.0]),
                 "xPyr": numpy.array([-6.0, 6.0]),
                 "xExc": numpy.array([-20.0, 20.0]),
                 "xInh": numpy.array([-500.0, 500.0])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh", "vInhf", "xInhf", "vInhs2f", "xInhs2f"),
        default=("vPyr", "vExc", "vInh", "xPyr", "vInhf"),
        doc="""This represents the default state-variables of this Model to be
                                    monitored. It can be overridden for each Monitor if desired. The 
                                    corresponding state-variable indices for this model are :math:`y0 = 0`,
                                    :math:`y1 = 1`, :math:`y2 = 2`, :math:`y3 = 3`, :math:`y4 = 4`, and
                                    :math:`y5 = 5`""")

    state_variables = ["vPyr", "vExc", "vInh", "xPyr", "xExc", "xInh", "vInhf", "xInhf", "vInhs2f", "xInhs2f" ]
    _nvar = 6
    cvar = numpy.array([1, 2, 6], dtype=numpy.int32)

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The dynamic equations were taken from:
        TODO: add equations and finish the model ...
        """

        vPyr = state_variables[0, :]
        vExc = state_variables[1, :]
        vInh = state_variables[2, :]
        xPyr = state_variables[3, :]
        xExc = state_variables[4, :]
        xInh = state_variables[5, :]
        # Additional inhibitory loop in Wedling (2002)
        vInhf = state_variables[6, :]
        xInhf = state_variables[7, :]
        vInhs2f = state_variables[6, :]
        xInhs2f = state_variables[7, :]

        # NOTE: This is assumed to be \sum_j u_kj * S[y_{1_j} - y_{2_j}]
        lrc = coupling[0, :]
        src = local_coupling * (vExc - vInh)

        ## Intrinsic input
        input = numpy.random.normal(self.p, self.sigma, (len(coupling[0]), 1))

        S_pyr = self.e0 / (1 + numpy.exp(self.r * (self.v0 - (vExc - vInh - vInhf))))
        S_exc = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2exc * self.c * vPyr)))
        S_inh = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inh * self.c * vPyr)))
        S_inhf = self.e0 / (1 + numpy.exp(self.r * (self.v0 - self.c_pyr2inhf * self.c * vPyr - self.c_inh2inhf * self.c * vInhs2f)))


        ## NOTE, for local couplings:
        # vExc, xExc excitatory interneurons - gate for intrinsic (p) and external (lrc, src) inputs.
        # vInh, xInh inhibitory interneurons; vInh_f, xInh_f fast interneurons
        # vPyr, xPyr pyramidal neurons

        dvPyr = xPyr
        dvExc = xExc
        dvInh = xInh
        dxPyr = self.He / self.tau_e * S_pyr - (2 * xPyr) / self.tau_e - vPyr / self.tau_e ** 2
        dxExc = self.He / self.tau_e * (input + self.c_exc2pyr * self.c * S_exc + lrc + src) - (2 * xExc) / self.tau_e - vExc / self.tau_e ** 2
        dxInh = self.Hi / self.tau_i * (self.c_inh2pyr * self.c * S_inh) - (2 * xInh) / self.tau_i - vInh / self.tau_i ** 2

        dvInhf = xInhf
        dxInhf = self.Hif / self.tau_if * (self.c_inhf2pyr * self.c * S_inhf) - (2 * xInhf) / self.tau_if - vInhf / self.tau_if ** 2
        dvInhs2f = xInhs2f
        dxInhs2f = self.Hi / self.tau_i * S_inh - (2 * xInh) / self.tau_i - vInh / self.tau_i ** 2

        derivative = numpy.array([dvPyr, dvExc, dvInh, dxPyr, dxExc, dxInh, dvInhf, dxInhf, dvInhs2f, dxInhs2f])

        return derivative