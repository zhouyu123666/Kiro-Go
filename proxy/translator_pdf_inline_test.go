package proxy

import (
	"strings"
	"testing"
)

// A minimal, real, parseable PDF (ReportLab-generated) whose text layer contains
// "The project code name is BLUE-FALCON-2024." Used to verify server-side PDF
// text extraction end-to-end without depending on any external tool at runtime.
const sampleParseablePDFBase64 = "JVBERi0xLjMKJZOMi54gUmVwb3J0TGFiIEdlbmVyYXRlZCBQREYgZG9jdW1lbnQgKG9wZW5zb3VyY2UpCjEgMCBvYmoKPDwKL0YxIDIgMCBSCj4+CmVuZG9iagoyIDAgb2JqCjw8Ci9CYXNlRm9udCAvSGVsdmV0aWNhIC9FbmNvZGluZyAvV2luQW5zaUVuY29kaW5nIC9OYW1lIC9GMSAvU3VidHlwZSAvVHlwZTEgL1R5cGUgL0ZvbnQKPj4KZW5kb2JqCjMgMCBvYmoKPDwKL0NvbnRlbnRzIDcgMCBSIC9NZWRpYUJveCBbIDAgMCA2MTIgNzkyIF0gL1BhcmVudCA2IDAgUiAvUmVzb3VyY2VzIDw8Ci9Gb250IDEgMCBSIC9Qcm9jU2V0IFsgL1BERiAvVGV4dCAvSW1hZ2VCIC9JbWFnZUMgL0ltYWdlSSBdCj4+IC9Sb3RhdGUgMCAvVHJhbnMgPDwKCj4+IAogIC9UeXBlIC9QYWdlCj4+CmVuZG9iago0IDAgb2JqCjw8Ci9QYWdlTW9kZSAvVXNlTm9uZSAvUGFnZXMgNiAwIFIgL1R5cGUgL0NhdGFsb2cKPj4KZW5kb2JqCjUgMCBvYmoKPDwKL0F1dGhvciAoYW5vbnltb3VzKSAvQ3JlYXRpb25EYXRlIChEOjIwMjYwNjE4MTE1NTQ0KzA4JzAwJykgL0NyZWF0b3IgKGFub255bW91cykgL0tleXdvcmRzICgpIC9Nb2REYXRlIChEOjIwMjYwNjE4MTE1NTQ0KzA4JzAwJykgL1Byb2R1Y2VyIChSZXBvcnRMYWIgUERGIExpYnJhcnkgLSBcKG9wZW5zb3VyY2VcKSkgCiAgL1N1YmplY3QgKHVuc3BlY2lmaWVkKSAvVGl0bGUgKHVudGl0bGVkKSAvVHJhcHBlZCAvRmFsc2UKPj4KZW5kb2JqCjYgMCBvYmoKPDwKL0NvdW50IDEgL0tpZHMgWyAzIDAgUiBdIC9UeXBlIC9QYWdlcwo+PgplbmRvYmoKNyAwIG9iago8PAovRmlsdGVyIFsgL0FTQ0lJODVEZWNvZGUgL0ZsYXRlRGVjb2RlIF0gL0xlbmd0aCAxNDAKPj4Kc3RyZWFtCkdhcFFoMEU9RiwwVVxIM1RccE5ZVF5RS2s/dGM+SVAsO1cjVTFeMjNpaFBFTV8/Q1c0S0lTaTwhWzdgI09CX3NLT0dtLm1KdGkxRWVEcTZyalNZNkFLdFBAWjwvJlAqIlhmcD09J1thVCsocCdmPnBCRnInMFQnRE89WihzL1dgZzo+UT5XPClzLn4+ZW5kc3RyZWFtCmVuZG9iagp4cmVmCjAgOAowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwNjEgMDAwMDAgbiAKMDAwMDAwMDA5MiAwMDAwMCBuIAowMDAwMDAwMTk5IDAwMDAwIG4gCjAwMDAwMDAzOTIgMDAwMDAgbiAKMDAwMDAwMDQ2MCAwMDAwMCBuIAowMDAwMDAwNzIxIDAwMDAwIG4gCjAwMDAwMDA3ODAgMDAwMDAgbiAKdHJhaWxlcgo8PAovSUQgCls8YTc5YThmNGJlNjA1YTU5MDA0MzIyZWVhZTdlYTI3YjI+PGE3OWE4ZjRiZTYwNWE1OTAwNDMyMmVlYWU3ZWEyN2IyPl0KJSBSZXBvcnRMYWIgZ2VuZXJhdGVkIFBERiBkb2N1bWVudCAtLSBkaWdlc3QgKG9wZW5zb3VyY2UpCgovSW5mbyA1IDAgUgovUm9vdCA0IDAgUgovU2l6ZSA4Cj4+CnN0YXJ0eHJlZgoxMDEwCiUlRU9GCg=="

func makePDFDoc(name, b64 string) KiroDocument {
	doc := KiroDocument{Name: name, Format: "pdf"}
	doc.Source.Bytes = b64
	return doc
}

func TestInlinePDFDocumentsExtractsTextAndRemovesPDF(t *testing.T) {
	docs := []KiroDocument{makePDFDoc("report.pdf", sampleParseablePDFBase64)}

	text, kept := inlinePDFDocuments("read the pdf", docs)

	if len(kept) != 0 {
		t.Fatalf("expected the extracted PDF to be removed from documents, got %d remaining", len(kept))
	}
	if !strings.Contains(text, "read the pdf") {
		t.Fatalf("expected original user text to be preserved, got %q", text)
	}
	if !strings.Contains(text, "BLUE-FALCON-2024") {
		t.Fatalf("expected extracted PDF text to be inlined, got %q", text)
	}
	if !strings.Contains(text, "report.pdf") {
		t.Fatalf("expected the document name label in the inlined block, got %q", text)
	}
}

func TestInlinePDFDocumentsPreservesUnparseablePDF(t *testing.T) {
	bad := makePDFDoc("broken.pdf", "JVBERi0xLjQgbm90LWEtcmVhbC1wZGY=") // "%PDF-1.4 not-a-real-pdf"
	docs := []KiroDocument{bad}

	text, kept := inlinePDFDocuments("hello", docs)

	if text != "hello" {
		t.Fatalf("expected text unchanged when extraction fails, got %q", text)
	}
	if len(kept) != 1 || kept[0].Name != "broken.pdf" {
		t.Fatalf("expected the unparseable PDF to be preserved, got %+v", kept)
	}
}

func TestInlinePDFDocumentsLeavesNonPDFUntouched(t *testing.T) {
	csv := KiroDocument{Name: "data.csv", Format: "csv"}
	csv.Source.Bytes = "Y29sMSxjb2wyCjEsMgo=" // col1,col2\n1,2\n
	docs := []KiroDocument{csv}

	text, kept := inlinePDFDocuments("look", docs)

	if text != "look" {
		t.Fatalf("expected text unchanged for non-PDF, got %q", text)
	}
	if len(kept) != 1 || kept[0].Format != "csv" {
		t.Fatalf("expected the CSV document untouched, got %+v", kept)
	}
}

func TestInlinePDFDocumentsMixedKeepsNonPDFDropsPDF(t *testing.T) {
	csv := KiroDocument{Name: "data.csv", Format: "csv"}
	csv.Source.Bytes = "Y29sMSxjb2wyCjEsMgo="
	docs := []KiroDocument{
		makePDFDoc("report.pdf", sampleParseablePDFBase64),
		csv,
	}

	text, kept := inlinePDFDocuments("", docs)

	if !strings.Contains(text, "BLUE-FALCON-2024") {
		t.Fatalf("expected extracted PDF text inlined, got %q", text)
	}
	if len(kept) != 1 || kept[0].Format != "csv" {
		t.Fatalf("expected only the CSV document to remain, got %+v", kept)
	}
}

func TestExtractPDFTextReturnsEmptyOnGarbage(t *testing.T) {
	if got := extractPDFText("!!!!not base64 at all!!!!"); got != "" {
		t.Fatalf("expected empty string for non-base64 input, got %q", got)
	}
	if got := extractPDFText(""); got != "" {
		t.Fatalf("expected empty string for empty input, got %q", got)
	}
}
