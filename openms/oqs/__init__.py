# Copyright 2023. Triad National Security, LLC. All rights reserved. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yu Zhang <zhy@lanl.gov>

from openms.lib import backend

class TDNEGF(object):

    def __init__(self, system, baths=None, dt=0.01, tmax=1.e2):
        '''
        sys:
        bath: [List]
        '''

        self.sys = system
        self.baths = baths
        self.dt = dt
        self.tmax = tmax


    def self_energies(self):
        '''
        evaluate self-energies due to sys-bath coupling

        '''

        return None


    def propagation(self):

        '''
        main driver of propagating the reduced density matrix
        '''

        return None


    def get_currents(self):
        '''
        return currents
        '''

        return None


