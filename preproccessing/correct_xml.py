import xml.etree.ElementTree as ET
import os

xml_list = []

for root, dirs, files in os.walk(r'./data'):
    for file in files:
        if file.endswith('.xml'):
            xml_list.append(os.path.join(root, file))

for xml_file in xml_list:
    # Open original file
    # et = xml.etree.ElementTree.parse(xml_file)

    tree = ET.parse(xml_file)
    root = tree.getroot()
    fname = root.find('filename')
    fname.text = fname.text+'.jpg'

    fname = root.find('filename')

    size = root.find('size')
    width = size.find('width')
    width.text = "640"

    height = size.find('height')
    height.text = "640"

    # Write back to file
    #et.write('file.xml')
    tree.write(xml_file, encoding="utf-8")