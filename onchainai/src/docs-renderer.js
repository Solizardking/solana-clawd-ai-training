import { marked } from 'marked';
import DOMPurify from 'dompurify';
window.marked = { parse: markdown => DOMPurify.sanitize(marked.parse(markdown)) };
