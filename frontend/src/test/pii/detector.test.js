import { describe, expect, it } from 'vitest';
import { scanPii } from '../../pii';

describe('scanPii', () => {
  it('detects global high-confidence identifiers without exposing snippets', () => {
    const result = scanPii('Email jane@example.org or call +1 202 555 0199.', { locale: 'en' });

    expect(result.summary.high.email).toBe(1);
    expect(result.summary.high.phone).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
    for (const finding of result.findings) {
      expect(Object.keys(finding).sort()).toEqual([
        'category',
        'confidence',
        'end',
        'id',
        'redactionToken',
        'ruleId',
        'start',
      ]);
      expect(finding).not.toHaveProperty('snippet');
      expect(finding).not.toHaveProperty('match');
    }
  });

  it('keeps finding IDs unique when phone rules run across locales', () => {
    const result = scanPii('Call +1 202 555 0199 or +31 20 123 4567.', { locale: 'en' });
    const ids = result.findings.map((finding) => finding.id);

    expect(new Set(ids).size).toBe(ids.length);
  });

  it.each([
    ['Hb 12.3 g/dl, Thrombozyten 250000/µl, CK 1250 U/l, Kreatinin 0.9 mg/dl.'],
    ['arr[GRCh37] 22q11.21(18912231_21465672)x1. OMIM 154700. chr2:166848646. PMID 12345678.'],
    ['Blutdruck 120/80 mmHg. Visus 0.4/0.5. 46,XY.'],
    ['St. p. Appendektomie 2010, aktuell Hb 12.5 normal.'],
    ['Dr. Meier sah den Patienten 2019 mit Krampfanfall und Fieber.'],
    ['Thrombozyten typically 250000/µl. Leukozyten im Mittel 12345678.'],
    ['Verlauf 01.03.2019 bis 05.06.2019 14 Tage. Ratio 0.123456789.'],
  ])('does not flag clinical values as high-confidence identifiers: %s', (text) => {
    for (const locale of ['en', 'de']) {
      expect(scanPii(text, { locale }).summary.high).toEqual({});
    }
  });

  it.each([
    ['Tel. 030 12345678'],
    ['Erreichbar unter 0171 1234567'],
    ['Call 202-555-0199'],
    ['Fax: 12345678'],
    ['Phone number: 202-555-0199'],
    ['Reachable at (202) 555-0199'],
    ['Tel.-Nr.: 12345678'],
  ])('still detects dialled phone numbers: %s', (text) => {
    expect(scanPii(text, { locale: 'de' }).summary.high.phone).toBe(1);
  });

  it('covers a dot-separated French phone number completely', () => {
    const text = 'Joignable au 01.23.45.67.89';
    const { findings } = scanPii(text, { locale: 'fr' });
    expect(findings).toHaveLength(1);
    expect(text.slice(findings[0].start, findings[0].end)).toBe('01.23.45.67.89');
  });

  it('detects a house number with a letter suffix', () => {
    const result = scanPii('Lives at 221B Baker Street.', { locale: 'en' });
    expect(result.summary.high.address).toBe(1);
  });

  it('detects an English street address', () => {
    const result = scanPii('Lives at 12 Main Street, Springfield.', { locale: 'en' });
    expect(result.summary.high.address).toBe(1);
  });

  it('detects English medical record labels', () => {
    const result = scanPii('MRN: AB-123456. Patient has seizures.', { locale: 'en' });
    expect(result.summary.high.medical_record).toBe(1);
  });

  it('detects German KVNR and DOB labels', () => {
    const result = scanPii('Geburtsdatum: 12.03.1980. Krankenversichertennummer A123456789.', {
      locale: 'de',
    });
    expect(result.summary.high.dob).toBe(1);
    expect(result.summary.review.date).toBeUndefined();
    expect(result.summary.high.national_identifier).toBe(1);
  });

  it('marks German titled person names as review confidence', () => {
    const result = scanPii('Herr Popp ist dumm', { locale: 'de' });

    expect(result.summary.review.person_name).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  it('marks configured titled person names even when the selected language differs', () => {
    const result = scanPii('M Dupont a des crises', { locale: 'en' });

    expect(result.summary.review.person_name).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  it.each([
    ['fr', 'M René a des crises'],
    ['es', 'Sr José tiene convulsiones'],
  ])('marks %s titled person names with accented letters', (locale, text) => {
    const result = scanPii(text, { locale });

    expect(result.summary.review.person_name).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  it.each([
    ['en', 'Mr Smith has seizures'],
    ['fr', 'M Dupont a des crises'],
    ['es', 'Sr Garcia tiene convulsiones'],
    ['nl', 'Dhr Jansen heeft aanvallen'],
  ])('marks %s titled person names as review confidence', (locale, text) => {
    const result = scanPii(text, { locale });

    expect(result.summary.review.person_name).toBe(1);
    expect(result.findings[0]).not.toHaveProperty('text');
  });

  // Untitled-name detection was removed: capitalised phenotype phrases are
  // indistinguishable from "Firstname Lastname" without a name lexicon.
  it.each([
    ['Global Developmental Delay, Atrial Septal Defect and Short Stature were noted.'],
    ['The patient has Intellectual Disability, Hearing Loss and Cleft Palate.'],
    ['Der Patient zeigt eine Globale Entwicklungsverzögerung.'],
    ['Hypoplasie der Lunge und Aplasie der Niere.'],
    ['Aortenwurzel Dilatation. Schwere Sprachentwicklungsverzögerung.'],
    ['Pectus Carinatum was noted.'],
    ['Beidseitige Ptosis und Generalisierte Anfälle.'],
    ['John Smith has seizures'],
    ['Patientin Anna Müller hat Fieber'],
  ])('does not mark capitalised phrases without a title as names: %s', (text) => {
    for (const locale of ['en', 'de']) {
      expect(scanPii(text, { locale }).summary.review.person_name).toBeUndefined();
    }
  });

  it('detects French NIR labels', () => {
    const result = scanPii('NIR 1900675123456 pour le patient.', { locale: 'fr' });
    expect(result.summary.high.national_identifier).toBe(1);
  });

  it('detects Spanish DNI/NIE and clinical history labels', () => {
    const result = scanPii('DNI 12345678Z. Número de historia clínica HC-778899.', {
      locale: 'es',
    });
    expect(result.summary.high.national_identifier).toBe(1);
    expect(result.summary.high.medical_record).toBe(1);
  });

  it('detects Dutch BSN and address context', () => {
    const result = scanPii('BSN 111222333. Adres: Hoofdstraat 12, 1234 AB Leiden.', {
      locale: 'nl',
    });
    expect(result.summary.high.national_identifier).toBe(1);
    expect(result.summary.high.address).toBe(1);
  });

  it('does not flag common HPO IDs or model identifiers', () => {
    const result = scanPii('HP:0001250 with model FremyCompany/BioLORD-2023-M.', { locale: 'en' });
    expect(result.hasFindings).toBe(false);
  });

  it('marks standalone exact dates as review confidence', () => {
    const result = scanPii('Follow-up discussed on 12/03/2024.', { locale: 'en' });
    expect(result.summary.review.date).toBe(1);
    expect(result.summary.high.date).toBeUndefined();
  });

  it('keeps address matches bounded to one line', () => {
    const text = 'Adres: Hoofdstraat 12, 1234 AB Leiden.\nSeizures described in detail.';
    const result = scanPii(text, { locale: 'nl' });
    const address = result.findings.find((finding) => finding.category === 'address');

    expect(address).toBeDefined();
    expect(address.end).toBeLessThan(text.indexOf('Seizures'));
  });
});
