
from django import template 
register = template.Library()


@register.simple_tag(takes_context=True)
def load_url_full(context, **kwargs):
    query = context['request'].GET.copy()
    for k,v in kwargs.items():
        query[k] = v
    return query.urlencode()
