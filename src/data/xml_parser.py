import xml.etree.ElementTree as ET
import os
from ..utils.point import Point


class XMLParser:
    """
    Reads a page XML file and extracts the baselines. The baselines are returned split into equidistant segments.
    If the baseline described in the page XML file contained a corner between two of these new equidistant corner
    points is is ignored and instead a baseline segment is saved where the last point before the original corner
    is connected to the first point after the corner.
    """
    def __init__(self, xml_filename: str):
        # Load the XML file
        self.tree = ET.parse(os.path.join(xml_filename))
        self.root = self.tree.getroot()

        # Extract name and size data
        self.filename = self.root.getchildren()[1].attrib['imageFilename']
        self.width = int(self.root.getchildren()[1].attrib['imageWidth'])
        self.height = int(self.root.getchildren()[1].attrib['imageHeight'])

        self.baselines = self.extract_points()
        self.scaled = False

    def extract_points(self) -> dict:
        """
        Extracts the text regions of the xml file as polygons and saves them in a dict.
        :return:    The dict where for every region type a list of polygons (that are again list of Points)
                    is given. The polygons are exactly the text regions described in the Page XML file.
        """
        baselines = []

        for region in self.root.getchildren()[1]:
            if 'TextRegion' in region.tag:
                for child in region:
                    if 'TextLine' in child.tag:
                        # One file (cPAS-2508.xml) contains a baseline without a 'Baseline' field.
                        if len(child.getchildren()) > 1:
                            baseline_string = child.getchildren()[1].attrib['points']
                            
                            baseline_points_string = baseline_string.split()

                            points = []
                            for p in baseline_points_string:
                                point = Point()
                                point.set_from_string(coords=p, sep=',')
                                points.append(point)
                            baselines.append(points)

        return baselines


    # def scale(self, max_side: int):
    #     """
    #     For images with at least one side larger than max_side scales the image and polygons
    #     such that the maximal side length is given by max_side.
    #     If max_side is larger than the maximum of width and height nothing is done.
    #     :param max_side: Maximally allowed side length
    #     """
    #     if self.scaled:
    #         return
    #     elif max(self.width, self.height) > max_side:
    #         ratio_w = float(max_side) / self.width
    #         ratio_h = float(max_side) / self.height
    #         w = self.width
    #         h = self.height
    #
    #         self.width = round(w * ratio_w)
    #         self.height = round(h * ratio_h)
    #
    #         for bl in self.baselines:
    #             for point in bl:
    #                 point.scale(ratio_w, ratio_h)
    #
    #         self.scaled = True

    def scale(self, min_side: int):
        """
        For images with at least one side larger than max_side scales the image and polygons
        such that the minimal side length is given by max_side.
        If max_side is larger than the maximum of width and height nothing is done.
        :param max_side: Maximally allowed side length
        """
        if self.scaled:
            return
        else:
            ratio = min_side / min(self.width, self.height)
            w = self.width
            h = self.height

            self.width = min_side if w < h else round(w * ratio)
            self.height = min_side if h < w else round(h * ratio)

            for region in self.baselines:
                for point in region:
                    point.scale(ratio, ratio)

            self.scaled = True

    def get_baselines(self):
        return self.baselines