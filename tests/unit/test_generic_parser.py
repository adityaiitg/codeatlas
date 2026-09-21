"""Unit tests for the generic multi-language parser."""

from pathlib import Path

from codeatlas.parser.generic_parser import GenericCodeParser


def test_parse_typescript_file(tmp_path: Path):
    ts_code = """
import { Request, Response } from 'express';
import AuthService from './auth.service';

export interface UserPayload {
    id: string;
    email: string;
}

export class AuthController {
    private service: AuthService;

    constructor() {
        this.service = new AuthService();
    }

    async login(req: Request, res: Response) {
        return this.service.authenticate(req.body);
    }
}

export const helperUtil = () => {
    return true;
};
"""
    file_path = tmp_path / "auth.controller.ts"
    file_path.write_text(ts_code, encoding="utf-8")

    parser = GenericCodeParser("typescript")
    symbols, chunks, edges = parser.parse_file(file_path)

    assert len(symbols) >= 3  # module, UserPayload, AuthController, helperUtil
    names = [s.name for s in symbols]
    assert "AuthController" in names
    assert "UserPayload" in names
    assert "helperUtil" in names

    assert len(chunks) >= 3
    assert len(edges) >= 2  # imports express, auth.service
    import_targets = [e.target_id for e in edges if e.edge_type.value == "IMPORTS"]
    assert any("express" in t for t in import_targets)
    assert any("auth.service" in t for t in import_targets)


def test_parse_go_file(tmp_path: Path):
    go_code = """
package main

import "fmt"

type Server struct {
    port int
}

func StartServer(port int) {
    fmt.Println("Server running")
}
"""
    file_path = tmp_path / "server.go"
    file_path.write_text(go_code, encoding="utf-8")

    parser = GenericCodeParser("go")
    symbols, chunks, edges = parser.parse_file(file_path)

    names = [s.name for s in symbols]
    assert "Server" in names
    assert "StartServer" in names
    assert len(chunks) >= 2
