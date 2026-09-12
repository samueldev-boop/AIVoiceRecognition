"""Algebra de intervalos: lo que antes eran merge() y overlap_len() a mano."""

from app import intervalos


def test_union_fusiona_solapes_y_ordena():
    u = intervalos.union([(2.5, 4.0), (1.0, 3.0), (10.0, 11.0)])
    assert intervalos.tramos(u) == [(1.0, 4.0), (10.0, 11.0)]
    assert intervalos.duracion(u) == 4.0


def test_union_descarta_segmentos_vacios_o_invertidos():
    u = intervalos.union([(1.0, 1.0), (3.0, 2.0), (5.0, 6.0)])
    assert intervalos.tramos(u) == [(5.0, 6.0)]


def test_solape_e_iou():
    a = intervalos.union([(0.0, 10.0)])
    b = intervalos.union([(5.0, 15.0)])
    assert intervalos.solape(a, b) == 5.0
    assert intervalos.iou(a, b) == 5.0 / 15.0
    assert intervalos.iou(a, a) == 1.0
    assert intervalos.iou(a, intervalos.union([(20.0, 30.0)])) == 0.0


def test_iou_de_vacios_no_divide_por_cero():
    vacio = intervalos.union([])
    assert intervalos.iou(vacio, vacio) == 0.0
    assert intervalos.duracion(vacio) == 0.0
