from jinja2 import Template

with open("./templates/mesherclass.j2", "rt") as f:
	text = f.read()

template = Template(text)

for position_type in [32, 64]:
	for label_type in [8, 16, 32, 64]:
		print(template.render(
			position_type=position_type, label_type=label_type
		))