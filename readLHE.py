import xml
import xml.etree.ElementTree
import numpy as np

class InitInfo():
    def __init__(self, init):
        for key, val in init.items():
            setattr(self, key, val)

class Event():
    def __init__(self, particles):
        self.particles = particles
        for p in particles:
            p.event = self

class Particle():
    def __init__(self, val_labels):
        for key, val in val_labels.items():
            setattr(self, key, val)
        self.p = [self.e, self.px, self.py, self.pz]

def NEvents(file):
    N = 0
    for event, element in xml.etree.ElementTree.iterparse(file):
        if element.tag == 'event':
            N += 1
    return N

def readEvent(file):
    labels = ['id', 'status', 'mother1', 'mother2', 'color1', 'color2', 'px', 'py', 'pz', 'e', 'm', 'time', 'spin']
    ev = []
    d_init = {}
    for _, block in xml.etree.ElementTree.iterparse(file):
        if block.tag == 'event':
            data = block.text.split('\n')[1:-1]
            val = {}
            part = []
            for p in data[1:]:
                line = [float(i) for i in p.split()]
                ind = np.arange(len(labels))
                for lin, i in zip(line,ind):
                    val[labels[i]] = lin
                part.append(Particle(val))
                val = {}
            ev.append(Event(part))
        if block.tag == 'init':
            d_init['beamA'] = float(block.text.split()[2])
            d_init['beamB'] = float(block.text.split()[3])
    init = InitInfo(d_init)
    return init, ev

def readInit(file):
    init = {}
    for _, block in xml.etree.ElementTree.iterparse(file):
        if block.tag == 'init':
            init['beamA'] = block.text.split()[2]
            init['beamB'] = block.text.split()[3]
    return init
